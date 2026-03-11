/**
 * app.js — Parallel tile pipeline for AI super-resolution
 * ════════════════════════════════════════════════════════════════════════════
 *
 * ┌─────────────────── ARCHITECTURE OVERVIEW ────────────────────────────┐
 * │                                                                       │
 * │  sourceCanvas (hidden, full input resolution)                        │
 * │       │                                                               │
 * │       ▼  generateTiles() → TileQueue (array of metadata)            │
 * │                                                                       │
 * │  ┌─── CONCURRENCY LIMITER (max N in-flight = pool size) ───────┐    │
 * │  │                                                               │    │
 * │  │  for each tile:                                              │    │
 * │  │    buildTileRGB() ──► WorkerPool.submit(tileId, buffer)     │    │
 * │  │                           ├─ Worker 0: tf.execute (GPU)     │    │
 * │  │                           ├─ Worker 1: tf.execute (GPU)  …  │    │
 * │  │                           └─ Worker N: tf.execute (GPU)     │    │
 * │  │    .then(outBuffer) ──► TileCompositor.receive()            │    │
 * │  │                           ├─ write to bandBuf (per-band)    │    │
 * │  │                           ├─ update previewCanvas (live)    │    │
 * │  │                           └─ if band complete:              │    │
 * │  │                                PNGStreamEncoder.feedBand()  │    │
 * │  └───────────────────────────────────────────────────────────────┘  │
 * │                                                                       │
 * │  PNGStreamEncoder.finalize() → Blob (Modo Blob) │ disk (Modo FSA)   │
 * └───────────────────────────────────────────────────────────────────────┘
 *
 * ════════════════════ MEMORY ANALYSIS (4000×4000 input) ══════════════════
 *
 *   Component              Size            Lifecycle
 *   ─────────────────────  ──────────────  ───────────────────────────────
 *   sourceCanvas backing   64 MB           lives until resetUI()
 *   previewCanvas          ~4 MB           lives until resetUI()
 *   per-band bandBuf       ~13 MB each     allocated/freed per band
 *   active band bufs       ~26 MB peak     at most 2 bands simultaneously
 *   in-flight RGB tiles    ~48 KB × N      ~300 KB for N=6
 *   in-flight out tiles    ~768 KB × N     ~4.6 MB for N=6
 *   PNG chunks (Blob mode) 50-400 MB       accumulates until finalize()
 *   PNG chunks (FSA mode)  ~0 MB           written to disk immediately
 *
 *   Peak RAM (Blob mode):  ~150-500 MB     vs ~2.3 GB for v2 architecture
 *   Peak RAM (FSA mode):   ~100 MB         safe for any image size
 *
 * ════════════════════════ PERFORMANCE GAINS ══════════════════════════════
 *
 *   v2 (sequential):  1 tile in-flight, GPU idles while CPU prepares next
 *   v3 (this file):   N tiles in-flight, CPU preparation of tile N+1
 *                     overlaps with GPU inference of tile N
 *
 *   Expected speedup: 2-4× on discrete GPU, 1.5-2× on integrated GPU.
 *   Main bottleneck shifts from "wait for GPU" to "GPU throughput".
 *
 * ════════════════════ TRADEOFFS AND REAL LIMITS ══════════════════════════
 *
 *   • Each worker loads its own copy of the model (~80 MB RAM + VRAM).
 *     6 workers = ~480 MB model overhead. Worth it for the pipelining gain.
 *   • Chrome limits WebGL contexts to ~16 per page. Capped at 6 workers.
 *   • GPU command queues are ultimately serialized by the driver; true
 *     parallel GPU execution depends on the hardware scheduler.
 *     The benefit on integrated GPUs is mostly CPU-side pipelining.
 *   • Workers that share a physical GPU still benefit from overlapping
 *     CPU overhead (tile building, data transfer, compositing).
 */

"use strict";

// ════════════════════════════════════════════════════════════════════════════
// §1  FIXED CONSTANTS
// ════════════════════════════════════════════════════════════════════════════

const TILE_IN       = 64;    // fixed model input size (px per side)
const OUT_IN        = 256;   // fixed model output size = TILE_IN × SCALE
const SCALE         = 4;     // upscale factor
const PREVIEW_MAX   = 960;   // max dimension of the on-screen preview canvas
const MAX_INPUT_DIM = 4000;  // reject inputs larger than this
const FSA_THRESHOLD = 1200;  // recommend FSA save for larger images
const BLOB_WARN_MB  = 150;   // warn in UI if PNG estimate exceeds this
const METRICS_HZ    = 400;   // ms between metrics panel refreshes

// ════════════════════════════════════════════════════════════════════════════
// §2  ADAPTIVE TILE PARAMETERS
// ════════════════════════════════════════════════════════════════════════════
/**
 * Tile padding controls seam quality vs throughput tradeoff.
 *
 * The model takes a 64×64 input and returns 256×256. We only USE the central
 * "effective" region of the output, discarding the padded border. Larger
 * padding → smoother seams (more overlap) but more tiles (lower speed).
 *
 *  PAD | EFF  | OUT_EFF | Tiles for 1000×1000  | Notes
 *  ────┼──────┼─────────┼──────────────────────┼────────────────
 *   8  |  48  |   192   |       441            | max quality
 *   6  |  52  |   208   |       400            | balanced (default)
 *   4  |  56  |   224   |       361            | fastest
 *
 * @returns {{ PAD, EFF, OUT_PAD, OUT_EFF }}
 */
function computeTileParams(srcW, srcH) {
  const maxDim = Math.max(srcW, srcH);
  const PAD    = maxDim >= 2000 ? 4 : maxDim >= 800 ? 6 : 8;
  const EFF    = TILE_IN - 2 * PAD;
  return { PAD, EFF, OUT_PAD: PAD * SCALE, OUT_EFF: EFF * SCALE };
}

// ════════════════════════════════════════════════════════════════════════════
// §3  WORKER POOL SIZE HEURISTIC
// ════════════════════════════════════════════════════════════════════════════
/**
 * Scales with hardware but caps at 6 for two reasons:
 *  1. Each worker loads ~80 MB of model weights → 6 workers = ~480 MB.
 *  2. Chrome limits WebGL contexts to ~16 per page; 6 is a safe margin.
 *
 * Minimum 2: even on low-end hardware, 2 workers gives meaningful pipelining
 * (one building/compositing while the other is on the GPU).
 */
function computePoolSize() {
  const hw = navigator.hardwareConcurrency || 4;
  if (hw >= 16) return 6;
  if (hw >= 8)  return 5;
  if (hw >= 4)  return 4;
  return 2;
}

// ════════════════════════════════════════════════════════════════════════════
// §4  TILE GENERATOR
// ════════════════════════════════════════════════════════════════════════════
/**
 * Pre-computes all tile metadata (no pixel data yet — just coordinates).
 * Memory cost: ~200 bytes × N tiles (negligible).
 *
 * Tiles are ordered row-major (left to right, top to bottom) so that bands
 * complete roughly in order, minimizing compositor buffering pressure.
 *
 * @returns {{ tiles: TileMeta[], cols: number, rows: number }}
 */
function generateTiles(srcW, srcH, tp) {
  const { EFF, OUT_EFF } = tp;
  const cols  = Math.ceil(srcW / EFF);
  const rows  = Math.ceil(srcH / EFF);
  const tiles = [];
  let id = 0;

  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const srcX    = col * EFF;
      const srcY    = row * EFF;
      const srcEffW = Math.min(EFF,     srcW - srcX);
      const srcEffH = Math.min(EFF,     srcH - srcY);
      tiles.push({
        id,
        row, col,
        srcX,  srcY,
        srcEffW, srcEffH,
        dstX:    srcX    * SCALE,
        dstY:    srcY    * SCALE,
        outEffW: srcEffW * SCALE,
        outEffH: srcEffH * SCALE,
      });
      id++;
    }
  }
  return { tiles, cols, rows };
}

// ════════════════════════════════════════════════════════════════════════════
// §5  TILE RGB BUILDER
// ════════════════════════════════════════════════════════════════════════════
/**
 * Reads a small region from sourceCanvas and assembles a Float32[64×64×3]
 * tensor input with edge-clamped padding for the model.
 *
 * MEMORY: reads only ~70×70 px = ~19 KB. The returned Float32Array (48 KB)
 * is immediately transferred (zero-copy) to a worker.
 *
 * EDGE CLAMPING: pixels outside the image bounds are replicated from the
 * nearest border pixel. This is the standard "constant" or "reflect" padding
 * strategy for SR networks — avoids introducing artificial color gradients
 * at image edges that convolution would otherwise amplify.
 *
 * @param {number} srcX  - effective tile origin X in source
 * @param {number} srcY  - effective tile origin Y in source
 * @param {number} srcW  - source image width
 * @param {number} srcH  - source image height
 * @param {number} PAD   - padding pixels around effective region
 * @returns {Float32Array} [TILE_IN × TILE_IN × 3]
 */
function buildTileRGB(srcX, srcY, srcW, srcH, PAD) {
  const padX = srcX - PAD;
  const padY = srcY - PAD;
  // Read only the available sub-region (avoid out-of-bounds getImageData)
  const readX = Math.max(0, padX);
  const readY = Math.max(0, padY);
  const readW = Math.max(1, Math.min(srcW, padX + TILE_IN) - readX);
  const readH = Math.max(1, Math.min(srcH, padY + TILE_IN) - readY);

  const patch = sourceCtx.getImageData(readX, readY, readW, readH);
  const pd    = patch.data;
  const rgb   = new Float32Array(TILE_IN * TILE_IN * 3);

  for (let r = 0; r < TILE_IN; r++) {
    const imgY = Math.min(Math.max(padY + r, 0), srcH - 1);
    const pr   = imgY - readY;
    for (let c = 0; c < TILE_IN; c++) {
      const imgX = Math.min(Math.max(padX + c, 0), srcW - 1);
      const pc   = imgX - readX;
      const si   = (pr * readW + pc) * 4;
      const di   = (r  * TILE_IN + c) * 3;
      rgb[di]     = pd[si]     / 255;
      rgb[di + 1] = pd[si + 1] / 255;
      rgb[di + 2] = pd[si + 2] / 255;
    }
  }
  return rgb;
}

// ════════════════════════════════════════════════════════════════════════════
// §6  WORKER POOL
// ════════════════════════════════════════════════════════════════════════════
/**
 * Manages a pool of Worker threads, each running worker.js.
 *
 * DISPATCH PROTOCOL:
 *   submit(tileId, rgbBuffer) → Promise<ArrayBuffer>
 *
 *   If a worker is free, the tile is dispatched immediately.
 *   If all workers are busy, the job is queued (internal FIFO).
 *   When any worker finishes, it pulls the next queued job.
 *
 * CONCURRENCY CONTROL:
 *   The caller (startProcessing) uses a semaphore to limit in-flight tiles
 *   to `pool.actualSize`. This prevents unbounded queue buildup — at most
 *   N tiles worth of RGB buffers exist simultaneously (N × 48 KB ≈ 300 KB).
 *
 * MODEL LOADING:
 *   Workers load the model in PARALLEL during init(). With browser HTTP
 *   caching, the first worker fetches the model from the CDN; subsequent
 *   workers get it from the local disk cache. Total init time ≈ single
 *   worker init time + N × GPU warmup (each warmup is ~100-300 ms on WebGL).
 */
class WorkerPool {
  constructor(size) {
    this._targetSize = size;
    this.actualSize  = 0;       // workers that successfully loaded the model
    this.backend     = "—";
    this._workers    = [];
    this._free       = [];      // indices of idle workers
    this._queue      = [];      // { tileId, buf, res, rej } waiting for a worker
    this._jobs       = new Map();// tileId → { res, rej, workerIdx }
    this._terminated = false;
  }

  /** Spawn all workers. Returns a promise that resolves with the GPU backend name. */
  init(onProgress) {
    return new Promise((resolve, reject) => {
      let readyCount = 0;
      let settledCount = 0;

      const settle = () => {
        settledCount++;
        if (settledCount === this._targetSize) {
          readyCount > 0
            ? resolve(this.backend)
            : reject(new Error("All workers failed to initialize"));
        }
      };

      for (let i = 0; i < this._targetSize; i++) {
        const w   = new Worker("/enhancer/worker.js")
        const idx = i;
        w._id     = idx;
        w._busy   = false;
        this._workers.push(w);

        w.onmessage = ({ data }) => {
          if (data.type === "ready") {
            readyCount++;
            this.actualSize++;
            if (this.backend === "—" && data.backend) this.backend = data.backend;
            this._free.push(idx);
            this._drain();
            onProgress?.(readyCount, this._targetSize, this.backend);
            settle();
            return;
          }

          if (data.type === "tile_done") {
            const job = this._jobs.get(data.tileId);
            if (job) {
              this._jobs.delete(data.tileId);
              w._busy = false;
              this._free.push(idx);
              this._drain();
              job.res(data.buffer);
            }
            return;
          }

          if (data.type === "error") {
            // Worker-reported error for a specific tile
            const job = this._jobs.get(data.tileId);
            if (job) {
              this._jobs.delete(data.tileId);
              w._busy = false;
              this._free.push(idx);
              this._drain();
              job.rej(new Error(data.message || "Worker tile error"));
            } else if (!this.actualSize) {
              // Error during init (model load failure)
              settle();
            }
          }
        };

        w.onerror = (err) => {
          console.error(`Worker ${idx} crashed:`, err);
          w._busy = false;
          // Reject any job assigned to this worker
          for (const [tileId, job] of this._jobs) {
            if (job.workerIdx === idx) {
              this._jobs.delete(tileId);
              job.rej(new Error("Worker crashed"));
            }
          }
          settle();
        };
      }
    });
  }

  /**
   * Submit a tile for inference. The Float32Array buffer is transferred
   * (zero-copy) to a worker. Returns a Promise<ArrayBuffer> (output RGB).
   */
  submit(tileId, rgbBuffer) {
    if (this._terminated) return Promise.reject(new Error("WORKER_TERMINATED"));
    return new Promise((res, rej) => {
      const job = { tileId, buf: rgbBuffer, res, rej };
      if (this._free.length > 0) {
        this._dispatch(job);
      } else {
        // Queue for the next available worker
        this._queue.push(job);
      }
    });
  }

  _dispatch({ tileId, buf, res, rej }) {
    const idx = this._free.pop();
    this._workers[idx]._busy = true;
    this._jobs.set(tileId, { res, rej, workerIdx: idx });
    this._workers[idx].postMessage(
      { type: "process_tile", tileId, buffer: buf },
      [buf]
    );
  }

  _drain() {
    while (this._queue.length > 0 && this._free.length > 0) {
      this._dispatch(this._queue.shift());
    }
  }

  /** Number of currently busy workers (for metrics display). */
  get busyCount() { return this._workers.filter(w => w._busy).length; }

  /** Terminate all workers immediately. Outstanding Promises are rejected. */
  terminate() {
    this._terminated = true;
    this._workers.forEach(w => w.terminate());
    this._workers = [];
    this._free    = [];
    const q = [...this._queue];
    this._queue = [];
    q.forEach(j => j.rej(new Error("WORKER_TERMINATED")));
    this._jobs.forEach(j => j.rej(new Error("WORKER_TERMINATED")));
    this._jobs.clear();
  }
}

// ════════════════════════════════════════════════════════════════════════════
// §7  TILE COMPOSITOR
// ════════════════════════════════════════════════════════════════════════════
/**
 * Receives completed tiles (out of order from the parallel pool) and:
 *   1. Writes them to per-band RGBA buffers (bandBuf).
 *   2. Updates the preview canvas immediately (live progressive rendering).
 *   3. Flushes complete bands to the PNG encoder IN ORDER.
 *
 * OUT-OF-ORDER HANDLING:
 *   Tiles within a band can complete in any order. The compositor tracks
 *   `received` count per band. When all columns of a band arrive, the band
 *   is considered "complete" and scheduled for flushing.
 *
 *   Flushing is strictly sequential (PNG rows must be encoded in order).
 *   A `_flushChain` promise ensures no band is flushed until all prior
 *   bands have been written to the encoder, even if later bands complete first.
 *
 * PER-BAND BUFFER ALLOCATION:
 *   Each band gets its own `Uint8Array[outW × bandH × 4]`. At most
 *   `ceil(poolSize / cols) + 1` bands are active simultaneously.
 *   For a 4000×4000 image (cols≈71) with poolSize=6: at most 2 active bands.
 *   Each band buffer: 16000 × 208 × 4 = ~13 MB. Peak: ~26 MB. ✓
 *
 * PREVIEW RENDERING:
 *   Each tile is drawn to the previewCanvas immediately upon arrival,
 *   regardless of band flush order. An OffscreenCanvas (tile-sized, ~170 KB)
 *   is created per tile, used for drawImage() scale+blit, then GC'd.
 *
 * ALPHA CHANNEL:
 *   The model only processes RGB. Alpha is reconstructed at compositing time
 *   by nearest-neighbor upscaling from the sourceCanvas. Reading alpha at
 *   compositing time (not pre-loading) keeps peak memory bounded.
 */
class TileCompositor {
  /**
   * @param {object} opts
   * @param {number}              opts.outW
   * @param {number}              opts.outH
   * @param {number}              opts.cols       - total columns in tile grid
   * @param {object}              opts.tp         - tile params (OUT_PAD, OUT_EFF)
   * @param {PNGStreamEncoder}    opts.encoder
   * @param {CanvasRenderingContext2D} opts.srcCtx - sourceCanvas context (for alpha)
   * @param {number}              opts.srcW
   * @param {number}              opts.srcH
   * @param {CanvasRenderingContext2D} opts.prevCtx - preview canvas context
   * @param {number}              opts.prevScale
   * @param {function}            opts.onTileComposited - called after each tile
   */
  constructor(opts) {
    this._outW       = opts.outW;
    this._outH       = opts.outH;
    this._cols       = opts.cols;
    this._tp         = opts.tp;       // { OUT_PAD, OUT_EFF }
    this._encoder    = opts.encoder;
    this._srcCtx     = opts.srcCtx;
    this._srcW       = opts.srcW;
    this._srcH       = opts.srcH;
    this._prevCtx    = opts.prevCtx;
    this._prevScale  = opts.prevScale;
    this._onTile     = opts.onTileComposited;

    // band state: row → { buf: Uint8Array, received: number, bandH: number }
    this._bands      = new Map();
    this._nextFlush  = 0;             // next band row index to flush
    this._aborted    = false;

    // Sequential flush chain: each band flush is chained as a .then() on this.
    // This guarantees encoder.feedBand() is never called out of order,
    // even when later bands complete before earlier ones.
    this._flushChain = Promise.resolve();
  }

  /**
   * Called by the main pipeline when a worker returns an output tile.
   * This runs on the main thread's microtask queue — no actual concurrency.
   *
   * @param {TileMeta}    tile
   * @param {Float32Array} outRgb - Float32[256×256×3] from model
   */
  receive(tile, outRgb) {
    if (this._aborted) return;
    const { OUT_EFF } = this._tp;

    // ── Ensure the band buffer exists ──────────────────────────────────────
    if (!this._bands.has(tile.row)) {
      const bandH = Math.min(OUT_EFF, this._outH - tile.row * OUT_EFF);
      this._bands.set(tile.row, {
        buf:      new Uint8Array(this._outW * bandH * 4),
        received: 0,
        bandH,
      });
    }
    const band = this._bands.get(tile.row);

    // ── Write tile RGBA to the band buffer ─────────────────────────────────
    this._writeTile(outRgb, tile, band);
    band.received++;

    // ── Update live preview (non-blocking, GPU-accelerated via drawImage) ──
    this._updatePreview(outRgb, tile);

    // ── If this band is now complete, schedule its flush ────────────────────
    // We chain the flush onto _flushChain to guarantee sequential ordering.
    // Bands that complete early are held in _bands until their turn arrives.
    if (band.received === this._cols) {
      this._flushChain = this._flushChain
        .then(() => this._flushPending())
        .catch(err => { if (!this._aborted) console.error("Band flush error:", err); });
    }

    this._onTile?.();
  }

  /** Flush all consecutive complete bands starting from _nextFlush. */
  async _flushPending() {
    while (!this._aborted) {
      const band = this._bands.get(this._nextFlush);
      if (!band || band.received < this._cols) break;

      await this._encoder.feedBand(band.buf, band.bandH);
      this._bands.delete(this._nextFlush);
      this._nextFlush++;
    }
  }

  /** Write the effective output region of a tile into the band buffer with alpha. */
  _writeTile(outRgb, tile, band) {
    const { OUT_PAD } = this._tp;
    const { dstX, outEffW, outEffH, srcX, srcY, srcEffW, srcEffH } = tile;

    // ── Read alpha for this tile from the source canvas ────────────────────
    // Reading per-tile (not the whole image upfront) keeps memory usage O(1).
    const aX = Math.min(srcX, this._srcW - 1);
    const aY = Math.min(srcY, this._srcH - 1);
    const aW = Math.max(1, Math.min(srcEffW, this._srcW - aX));
    const aH = Math.max(1, Math.min(srcEffH, this._srcH - aY));
    const alpha = this._srcCtx.getImageData(aX, aY, aW, aH).data;

    const stride = this._outW * 4; // band buffer row stride in bytes

    for (let r = 0; r < outEffH; r++) {
      const srcR = r + OUT_PAD; // row in the full 256-px output tile
      for (let c = 0; c < outEffW; c++) {
        const srcC = c + OUT_PAD;
        const si   = (srcR * OUT_IN + srcC) * 3;         // float index in outRgb
        const di   = r * stride + (dstX + c) * 4;        // byte index in band buf

        // Clamp to [0, 255]; SR models can slightly overshoot
        band.buf[di]     = Math.max(0, Math.min(255, outRgb[si]     * 255 + 0.5));
        band.buf[di + 1] = Math.max(0, Math.min(255, outRgb[si + 1] * 255 + 0.5));
        band.buf[di + 2] = Math.max(0, Math.min(255, outRgb[si + 2] * 255 + 0.5));

        // Nearest-neighbor alpha upscale: map output pixel → source pixel
        const aX2 = Math.min(Math.floor(c / SCALE), aW - 1);
        const aY2 = Math.min(Math.floor(r / SCALE), aH - 1);
        band.buf[di + 3] = alpha[(aY2 * aW + aX2) * 4 + 3];
      }
    }
  }

  /** Draw this tile's output region to the (scaled) preview canvas. */
  _updatePreview(outRgb, tile) {
    const { OUT_PAD }           = this._tp;
    const { dstX, dstY, outEffW, outEffH } = tile;

    // Build a temporary RGBA for the effective region
    const tmpRGBA = new Uint8ClampedArray(outEffW * outEffH * 4);
    for (let r = 0; r < outEffH; r++) {
      const srcR = r + OUT_PAD;
      for (let c = 0; c < outEffW; c++) {
        const srcC = c + OUT_PAD;
        const si   = (srcR * OUT_IN + srcC) * 3;
        const di   = (r * outEffW + c) * 4;
        tmpRGBA[di]     = Math.max(0, Math.min(255, outRgb[si]     * 255 + 0.5));
        tmpRGBA[di + 1] = Math.max(0, Math.min(255, outRgb[si + 1] * 255 + 0.5));
        tmpRGBA[di + 2] = Math.max(0, Math.min(255, outRgb[si + 2] * 255 + 0.5));
        tmpRGBA[di + 3] = 255; // preview alpha always opaque (speed)
      }
    }

    // OffscreenCanvas: lives ~1ms, GC'd immediately after drawImage.
    // GPU-accelerated blit with bilinear downscaling to the preview size.
    const osc = new OffscreenCanvas(outEffW, outEffH);
    osc.getContext("2d").putImageData(new ImageData(tmpRGBA, outEffW, outEffH), 0, 0);

    const s  = this._prevScale;
    const px = Math.floor(dstX  * s);
    const py = Math.floor(dstY  * s);
    const pw = Math.ceil(outEffW * s);
    const ph = Math.ceil(outEffH * s);
    this._prevCtx.drawImage(osc, 0, 0, outEffW, outEffH, px, py, pw, ph);
  }

  /** Promise that resolves when all bands have been flushed to the encoder. */
  get done() { return this._flushChain; }

  abort() { this._aborted = true; }
}

// ════════════════════════════════════════════════════════════════════════════
// §8  METRICS TRACKER
// ════════════════════════════════════════════════════════════════════════════
/**
 * Tracks tiles/sec using a rolling 3-second window.
 * Rolling window avoids startup bias (first tiles are slow during GPU warmup).
 */
class MetricsTracker {
  constructor() {
    this._ts       = [];    // completion timestamps (ms)
    this._windowMs = 3000;  // rolling window duration
    this.total     = 0;
    this.done      = 0;
    this._t0       = null;
  }

  start(total) {
    this.total = total;
    this.done  = 0;
    this._ts   = [];
    this._t0   = performance.now();
  }

  record() {
    this.done++;
    const now = performance.now();
    this._ts.push(now);
    // Prune timestamps outside the rolling window
    const cutoff = now - this._windowMs;
    while (this._ts.length > 0 && this._ts[0] < cutoff) this._ts.shift();
  }

  /** Rolling tiles/second. Returns 0 if insufficient data. */
  get tilesPerSec() {
    if (this._ts.length < 2) return this.done > 0 ? (this.done / this.elapsed) : 0;
    const span = (performance.now() - this._ts[0]) / 1000;
    return span > 0 ? this._ts.length / span : 0;
  }

  /** ETA in seconds. Null if not yet calculable. */
  get etaSec() {
    const tps = this.tilesPerSec;
    return tps > 0 ? (this.total - this.done) / tps : null;
  }

  get elapsed() {
    return this._t0 ? (performance.now() - this._t0) / 1000 : 0;
  }

  get pct() {
    return this.total > 0 ? Math.round(100 * this.done / this.total) : 0;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// §9  PNG STREAM ENCODER
// ════════════════════════════════════════════════════════════════════════════
/**
 * Encodes the full-resolution output image as a PNG in a streaming fashion,
 * one horizontal band of rows at a time. Uses pako (zlib/deflate) for
 * compression, avoiding any giant canvas or toDataURL().
 *
 * Two modes:
 *   • Blob mode (default): chunks accumulate in memory as a Blob (no canvas).
 *     Safe for typical images. Memory = compressed PNG size (~50-400 MB for
 *     large inputs). Blob lives outside the JS heap (browser-managed).
 *   • FSA mode: each IDAT chunk is written to disk immediately via
 *     FileSystemWritableFileStream. Memory = one band buffer at a time (~13 MB).
 *     Supports arbitrarily large images on any hardware.
 *
 * WHY NOT toDataURL / canvas.toBlob:
 *   These require a fully-allocated canvas at output resolution (1 GB for
 *   16000×16000). This encoder never allocates anything at that scale.
 *
 * WHY NOT CompressionStream (native browser API):
 *   CompressionStream doesn't expose Z_SYNC_FLUSH, which is required to
 *   produce valid interleaved IDAT chunks without closing the deflate stream.
 *   Pako exposes granular flush control, making per-band IDAT emission possible.
 */
class PNGStreamEncoder {
  constructor(width, height) {
    this.w          = width;
    this.h          = height;
    this._chunks    = [];    // Blob mode: accumulated Uint8Array IDAT chunks
    this._writable  = null;  // FSA mode: FileSystemWritableFileStream
    this._pending   = [];    // pako output buffers not yet wrapped in chunks
    this._finalized = false;
    this._filterRow = new Uint8Array(1); // reused [0x00] filter byte

    // pako Deflate level 3: good compression/speed ratio for SR output.
    // Level 6 adds ~2× CPU time for <15% compression gain on photographic
    // content (which is what SR output looks like).
    this._z = new pako.Deflate({ level: 3 });
    this._z.onData = buf => this._pending.push(new Uint8Array(buf));
  }

  async initFSA(writable) {
    this._writable = writable;
    await writable.write(PNG_SIG);
    await writable.write(_pngIHDR(this.w, this.h));
  }

  initBlob() {
    this._chunks.push(PNG_SIG);
    this._chunks.push(_pngIHDR(this.w, this.h));
  }

  /**
   * Feed a band of `bandH` rows to the encoder.
   * @param {Uint8Array} rgba   - outW × bandH × 4 bytes, row-major
   * @param {number}     bandH  - number of rows in this band
   */
  async feedBand(rgba, bandH) {
    const stride = this.w * 4;
    for (let r = 0; r < bandH; r++) {
      this._z.push(this._filterRow, false);                              // filter byte
      this._z.push(rgba.subarray(r * stride, (r + 1) * stride), false); // row pixels
    }
    // Z_SYNC_FLUSH (=2): flush compressed bytes without closing the stream.
    // Produces one or more IDAT-ready data blobs in this._pending.
    this._z.push(new Uint8Array(0), 2);
    await this._flush();
  }

  async _flush() {
    for (const buf of this._pending) {
      if (!buf.length) continue;
      const idat = _pngChunk("IDAT", buf);
      if (this._writable) await this._writable.write(idat);
      else                this._chunks.push(idat);
    }
    this._pending = [];
  }

  /** Close the deflate stream and write IEND. Returns Blob in Blob mode, null in FSA mode. */
  async finalize() {
    if (this._finalized) return null;
    this._finalized = true;
    this._z.push(new Uint8Array(0), true); // Z_FINISH: writes adler32 checksum
    await this._flush();

    const iend = _pngChunk("IEND", new Uint8Array(0));
    if (this._writable) {
      await this._writable.write(iend);
      await this._writable.close();
      this._writable = null;
      return null; // already on disk
    }
    this._chunks.push(iend);
    return new Blob(this._chunks, { type: "image/png" });
  }

  abort() {
    if (this._finalized) return;
    this._finalized = true;
    try { this._z.push(new Uint8Array(0), true); } catch (_) {}
    this._pending = [];
    this._chunks  = [];
    if (this._writable) { this._writable.abort(); this._writable = null; }
  }
}

// ─── PNG utilities ────────────────────────────────────────────────────────────
const _CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let j = 0; j < 8; j++) c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
    t[i] = c;
  }
  return t;
})();

function _crc32(buf, s, e) {
  let crc = 0xFFFFFFFF;
  for (let i = s; i < e; i++) crc = _CRC_TABLE[(crc ^ buf[i]) & 0xFF] ^ (crc >>> 8);
  return (crc ^ 0xFFFFFFFF) >>> 0;
}

function _pngChunk(type, data) {
  const len  = data.length;
  const out  = new Uint8Array(12 + len);
  const view = new DataView(out.buffer);
  view.setUint32(0, len, false);
  out[4]=type.charCodeAt(0); out[5]=type.charCodeAt(1);
  out[6]=type.charCodeAt(2); out[7]=type.charCodeAt(3);
  if (len) out.set(data, 8);
  view.setUint32(8 + len, _crc32(out, 4, 8 + len), false);
  return out;
}

function _pngIHDR(w, h) {
  const d = new Uint8Array(13);
  const v = new DataView(d.buffer);
  v.setUint32(0, w, false); v.setUint32(4, h, false);
  d[8]=8; d[9]=6; // 8-bit RGBA
  return _pngChunk("IHDR", d);
}

const PNG_SIG  = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);

// ════════════════════════════════════════════════════════════════════════════
// §10  DOM REFERENCES
// ════════════════════════════════════════════════════════════════════════════
const fileInput       = document.getElementById("fileInput");
const dropZone        = document.getElementById("dropZone");
const originalPreview = document.getElementById("originalPreview");
const previewCanvas   = document.getElementById("previewCanvas");
const processBtn      = document.getElementById("processBtn");
const downloadBtn     = document.getElementById("downloadBtn");
const saveFSABtn      = document.getElementById("saveFSABtn");
const clearBtn        = document.getElementById("clearBtn");
const statusText      = document.getElementById("statusText");
const progressBar     = document.getElementById("progressBar");
const progressFill    = document.getElementById("progressFill");
const imgInfoOrig     = document.getElementById("imgInfoOriginal");
const imgInfoResult   = document.getElementById("imgInfoResult");
const previewScaleTag = document.getElementById("previewScaleTag");

// Metrics panel elements
const mWorkers  = document.getElementById("mWorkers");
const mSpeed    = document.getElementById("mSpeed");
const mTiles    = document.getElementById("mTiles");
const mEta      = document.getElementById("mEta");
const mBackend  = document.getElementById("mBackend");
const mPad      = document.getElementById("mPad");
const mMemEst   = document.getElementById("mMemEst");

const previewCtx = previewCanvas.getContext("2d");

// ════════════════════════════════════════════════════════════════════════════
// §11  SOURCE CANVAS (hidden, full input resolution)
// ════════════════════════════════════════════════════════════════════════════
// willReadFrequently: true → browser keeps a CPU-accessible copy without
// GPU readback. Critical for the thousands of getImageData calls in the pipeline.
const sourceCanvas = document.createElement("canvas");
const sourceCtx    = sourceCanvas.getContext("2d", { willReadFrequently: true });

// ════════════════════════════════════════════════════════════════════════════
// §12  APPLICATION STATE
// ════════════════════════════════════════════════════════════════════════════
let currentImage     = null;
let currentFileName  = "imagen";
let pool             = null;   // active WorkerPool (persists between jobs if not cancelled)
let poolReady        = false;
let isProcessing     = false;
let cancelRequested  = false;
let resultBlob       = null;   // last completed PNG Blob (Blob mode)
let metrics          = new MetricsTracker();
let metricsInterval  = null;

// ════════════════════════════════════════════════════════════════════════════
// §13  INITIALIZATION
// ════════════════════════════════════════════════════════════════════════════
startPool();
bindEvents();

async function startPool() {
  poolReady = false;
  processBtn.disabled = true;
  setStatus("Inicializando worker pool...");
  setModelDot("loading");

  const size = computePoolSize();
  pool = new WorkerPool(size);

  try {
    await pool.init((ready, total, backend) => {
      setStatus(`Cargando modelos: ${ready}/${total} workers (${backend})...`);
      if (mWorkers) mWorkers.textContent = `${ready}/${total}`;
      if (mBackend) mBackend.textContent = backend || "—";
    });
    poolReady = true;
    setModelDot("ready");
    if (mWorkers) mWorkers.textContent = `${pool.actualSize}/${size}`;
    if (mBackend) mBackend.textContent = pool.backend;
    processBtn.disabled = !currentImage;
    setStatus(currentImage ? "Imagen lista. Pulsa Procesar ×4." : `Worker pool listo (${pool.actualSize} workers · ${pool.backend}). Esperando imagen...`);
  } catch (err) {
    setStatus("Error iniciando workers: " + err.message, true);
  }
}

// ════════════════════════════════════════════════════════════════════════════
// §14  EVENT HANDLERS
// ════════════════════════════════════════════════════════════════════════════
function bindEvents() {
  fileInput.addEventListener("change", e => loadImageFromFile(e.target.files[0]));
  processBtn.addEventListener("click",  () => startProcessing(null));
  downloadBtn.addEventListener("click",  triggerDownload);
  saveFSABtn?.addEventListener("click",  triggerFSASave);
  clearBtn.addEventListener("click",    onClearOrCancel);

  dropZone.addEventListener("dragover",  e => { e.preventDefault(); dropZone.classList.add("drag-over"); });
  dropZone.addEventListener("dragleave", ()  => dropZone.classList.remove("drag-over"));
  dropZone.addEventListener("drop", e => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const f = e.dataTransfer.files[0];
    if (f?.type.startsWith("image/")) loadImageFromFile(f);
  });

  dropZone.addEventListener("click",  () => { if (!currentImage) fileInput.click(); });
  document.getElementById("selectFileBtn")?.addEventListener("click", e => { e.stopPropagation(); fileInput.click(); });
  document.getElementById("changeImgBtn")?.addEventListener("click",  e => { e.stopPropagation(); fileInput.click(); });
}

// ════════════════════════════════════════════════════════════════════════════
// §15  IMAGE LOADING
// ════════════════════════════════════════════════════════════════════════════
function loadImageFromFile(file) {
  if (!file) return;
  const blobUrl = URL.createObjectURL(file);
  const img = new Image();
  img.onerror = () => { URL.revokeObjectURL(blobUrl); setStatus("No se pudo cargar la imagen.", true); };
  img.onload = function () {
    URL.revokeObjectURL(blobUrl);
    const { width: w, height: h } = this;

    if (w > MAX_INPUT_DIM || h > MAX_INPUT_DIM) {
      setStatus(`Imagen demasiado grande (${w}×${h}). Máx: ${MAX_INPUT_DIM}×${MAX_INPUT_DIM} px.`, true);
      dropZone.classList.add("shake");
      dropZone.addEventListener("animationend", () => dropZone.classList.remove("shake"), { once: true });
      return;
    }

    currentFileName = file.name.replace(/\.[^/.]+$/, "");
    currentImage    = this;
    resultBlob      = null;

    // Copy full image to sourceCanvas for tile reads
    sourceCanvas.width  = w;
    sourceCanvas.height = h;
    sourceCtx.clearRect(0, 0, w, h);
    sourceCtx.drawImage(this, 0, 0);

    // Preview of original
    const pvUrl = URL.createObjectURL(file);
    originalPreview.src    = pvUrl;
    originalPreview.onload = originalPreview.onerror = () => URL.revokeObjectURL(pvUrl);
    dropZone.classList.add("has-image");

    clearResultArea();

    const outW = w * SCALE;
    const outH = h * SCALE;
    const tp   = computeTileParams(w, h);
    const cols = Math.ceil(w / tp.EFF);
    const rows = Math.ceil(h / tp.EFF);
    const tot  = cols * rows;
    const mb   = estimatePNGMB(w, h);

    imgInfoOrig.textContent   = `${w} × ${h} px`;
    imgInfoResult.textContent = `${outW} × ${outH} px (estimado)`;
    if (mMemEst) mMemEst.textContent = `~${mb} MB PNG`;
    if (mPad)    mPad.textContent    = `pad=${tp.PAD} · ${tot} tiles`;

    processBtn.disabled  = !poolReady;
    downloadBtn.disabled = true;
    updateFSAButton(w, h);

    setStatus(poolReady
      ? `Imagen lista (${w}×${h} px, ${tot} tiles, pad=${tp.PAD}). Pulsa Procesar ×4.`
      : "Imagen cargada. Esperando que el pool termine de cargar..."
    );
  };
  img.src = blobUrl;
}

// ════════════════════════════════════════════════════════════════════════════
// §16  MAIN PROCESSING PIPELINE
// ════════════════════════════════════════════════════════════════════════════
/**
 * Orchestrates the full pipeline:
 *   1. Compute adaptive tile parameters.
 *   2. Generate tile metadata grid.
 *   3. Initialize PNG encoder (Blob or FSA mode).
 *   4. Initialize TileCompositor.
 *   5. Run the parallel dispatch loop:
 *        for each tile:
 *          if at concurrency limit → await Promise.race(inFlight)
 *          build RGB tile → pool.submit()
 *          .then → compositor.receive()
 *   6. Drain remaining in-flight tiles.
 *   7. Await compositor flush chain (sequential band encoding).
 *   8. Finalize encoder → Blob or close file.
 *
 * CONCURRENCY LIMITER (semaphore pattern):
 *   We track a Set of in-flight Promises (`inFlight`).
 *   Before submitting a new tile, if `inFlight.size >= CONCURRENCY`,
 *   we await `Promise.race([...inFlight])` which resolves as soon as
 *   any tile completes (and removes itself from `inFlight` via .finally).
 *   This guarantees:
 *   • At most N tiles are being built/transferred/processed simultaneously.
 *   • RGB tile buffers (48 KB × N) are the only large pre-allocated data.
 *   • Workers are kept maximally busy (never idle waiting for the main thread).
 *
 * @param {FileSystemWritableFileStream|null} fsaWritable
 */
async function startProcessing(fsaWritable) {
  if (!currentImage || !poolReady || isProcessing) return;

  const srcW = sourceCanvas.width;
  const srcH = sourceCanvas.height;
  const outW = srcW * SCALE;
  const outH = srcH * SCALE;

  // ── Adaptive tile parameters ──────────────────────────────────────────────
  const tp = computeTileParams(srcW, srcH);
  const { tiles, cols, rows } = generateTiles(srcW, srcH, tp);
  const CONCURRENCY = pool.actualSize; // tiles in-flight simultaneously

  // ── Preview canvas ────────────────────────────────────────────────────────
  const prevScale = Math.min(1, PREVIEW_MAX / Math.max(outW, outH));
  const prevW     = Math.max(1, Math.ceil(outW * prevScale));
  const prevH     = Math.max(1, Math.ceil(outH * prevScale));
  previewCanvas.width  = prevW;
  previewCanvas.height = prevH;
  previewCtx.clearRect(0, 0, prevW, prevH);
  showPreviewCanvas(true);

  if (previewScaleTag) {
    const pct = Math.round(prevScale * 100);
    previewScaleTag.textContent = prevScale < 1 ? `preview ${pct}%` : "preview 1:1";
  }
  imgInfoResult.textContent = `${outW} × ${outH} px`;

  // ── PNG Encoder ───────────────────────────────────────────────────────────
  const encoder = new PNGStreamEncoder(outW, outH);
  if (fsaWritable) await encoder.initFSA(fsaWritable);
  else             encoder.initBlob();

  // ── Compositor ────────────────────────────────────────────────────────────
  const compositor = new TileCompositor({
    outW, outH, cols: cols, tp,
    encoder, srcCtx: sourceCtx, srcW, srcH,
    prevCtx: previewCtx, prevScale,
    onTileComposited: () => metrics.record(),
  });

  // ── Metrics ───────────────────────────────────────────────────────────────
  metrics.start(tiles.length);
  startMetricsUI(pool, metrics, cols, tp.PAD, tiles.length);

  setProcessing(true);
  cancelRequested = false;
  resultBlob      = null;
  setProgress(0);
  setStatus(`Procesando ${tiles.length} tiles con ${CONCURRENCY} workers en paralelo...`);

  // ── Parallel dispatch loop ────────────────────────────────────────────────
  const inFlight = new Set(); // active Promise<void> handles

  try {
    for (const tile of tiles) {

      // ── Check cancellation before each tile ──────────────────────────────
      if (cancelRequested) { break; }

      // ── Concurrency semaphore: wait for a slot ────────────────────────────
      // Promise.race resolves when any in-flight promise resolves.
      // The .finally inside each promise removes it from the Set BEFORE
      // race resolves, so by the time we re-check the while condition,
      // inFlight.size is already decremented. ✓
      while (inFlight.size >= CONCURRENCY) {
        await Promise.race([...inFlight]);
      }

      if (cancelRequested) break;

      // ── Build RGB tile (synchronous, ~0.5ms) ─────────────────────────────
      const rgb = buildTileRGB(tile.srcX, tile.srcY, srcW, srcH, tp.PAD);

      // ── Dispatch to pool, chain compositor.receive ───────────────────────
      // `p` is the full promise chain including .finally. Adding `p` to
      // inFlight ensures we track the complete lifecycle, not just the submit.
      const p = pool.submit(tile.id, rgb.buffer)
        .then(outBuffer => {
          if (!cancelRequested) {
            compositor.receive(tile, new Float32Array(outBuffer));
          }
        })
        .catch(err => {
          if (!cancelRequested) {
            console.warn(`Tile ${tile.id} failed:`, err.message);
            // Non-fatal: the tile will be missing from the output but
            // processing continues. Alternative: re-throw to abort all.
          }
        })
        .finally(() => inFlight.delete(p));

      inFlight.add(p);
    }

    // ── Drain: wait for all remaining in-flight tiles ─────────────────────
    await Promise.allSettled([...inFlight]);
    inFlight.clear();

    if (cancelRequested) throw new Error("CANCELLED");

    // ── Wait for compositor to finish flushing all bands to encoder ────────
    // By the time Promise.allSettled resolves, all compositor.receive() calls
    // have been made (they happen in .then, which runs before .finally).
    // But feedBand() is async; this await ensures the encoder receives everything.
    await compositor.done;

    // ── Finalize PNG ──────────────────────────────────────────────────────
    const blob = await encoder.finalize();
    resultBlob  = blob;

    const elap = metrics.elapsed.toFixed(1);
    const tps  = metrics.tilesPerSec.toFixed(1);
    const mode = fsaWritable ? "guardado en disco" : "listo";
    setProgress(100);
    setStatus(`Listo en ${elap}s · ${tps} t/s · ${outW}×${outH} px · ${pool.backend} · ${mode}`);

    if (resultBlob) {
      const mb = Math.round(resultBlob.size / (1024 * 1024));
      downloadBtn.textContent = `Descargar PNG (${mb} MB)`;
      downloadBtn.disabled    = false;
    } else {
      downloadBtn.textContent = "Guardado en disco";
      downloadBtn.disabled    = true;
    }
    setProcessing(false);

  } catch (err) {
    compositor.abort();
    encoder.abort();
    inFlight.clear();

    if (err.message === "CANCELLED") {
      setStatus("Procesamiento cancelado.");
      setProgress(0);
    } else {
      setStatus("Error en el pipeline: " + err.message, true);
      console.error(err);
    }
    setProcessing(false);
  } finally {
    stopMetricsUI();
  }
}

// ════════════════════════════════════════════════════════════════════════════
// §17  EXPORT FUNCTIONS
// ════════════════════════════════════════════════════════════════════════════

/** Download the finished PNG via an Object URL (no base64, no toDataURL). */
function triggerDownload() {
  if (!resultBlob) return;
  const url = URL.createObjectURL(resultBlob);
  const a   = document.createElement("a");
  a.href     = url;
  a.download = `${currentFileName}_ai_x4_Novage.png`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 90_000);
}

/**
 * File System Access API export.
 *
 * TWO MODES depending on state:
 *
 *   A) Post-processing (resultBlob exists): write the finished Blob to disk.
 *      Useful for freeing RAM after a Blob-mode job.
 *
 *   B) Pre-processing (no resultBlob, image loaded): open the file dialog
 *      FIRST (requires user gesture), then start processing with streaming
 *      writes. This is the memory-optimal path for very large images.
 *      PNG is written to disk band-by-band, peak memory ≈ one band (~13 MB).
 *
 * LIMITATION: showSaveFilePicker() is only available in Chrome 86+ and Edge.
 * Safari has limited support as of 2024.
 */
async function triggerFSASave() {
  if (!("showSaveFilePicker" in window)) {
    setStatus("File System Access API no disponible (requiere Chrome 86+).", true);
    return;
  }
  let handle;
  try {
    handle = await window.showSaveFilePicker({
      suggestedName: `${currentFileName}_ai_x4_Novage.png`,
      types: [{ description: "PNG Image", accept: { "image/png": [".png"] } }],
    });
  } catch (e) {
    if (e.name !== "AbortError") setStatus("Error al abrir diálogo: " + e.message, true);
    return;
  }

  if (resultBlob) {
    // Mode A: write existing blob to disk
    try {
      const wr = await handle.createWritable();
      await wr.write(resultBlob);
      await wr.close();
      setStatus(`Guardado en disco. ${Math.round(resultBlob.size/1024/1024)} MB`);
    } catch (e) { setStatus("Error al guardar: " + e.message, true); }
  } else if (currentImage && poolReady && !isProcessing) {
    // Mode B: stream processing directly to disk
    try {
      const wr = await handle.createWritable();
      await startProcessing(wr);
    } catch (e) { setStatus("Error en FSA streaming: " + e.message, true); }
  }
}

// ════════════════════════════════════════════════════════════════════════════
// §18  CANCEL / CLEAR
// ════════════════════════════════════════════════════════════════════════════
function onClearOrCancel() {
  if (isProcessing) {
    cancelRequested = true;
    // Terminate the current pool: in-flight submissions reject immediately,
    // unblocking the concurrency-limiter await and exiting the for loop.
    if (pool) { pool.terminate(); pool = null; }
    poolReady = false;
    processBtn.disabled = true;
    clearBtn.textContent = "Limpiar";
    clearResultArea();
    setProgress(0);
    setProcessing(false);
    setStatus("Cancelado. Reiniciando worker pool...");
    // Re-initialize pool so the user can process again immediately
    startPool();
  } else {
    resetUI();
  }
}

function clearResultArea() {
  previewCanvas.width  = 0;
  previewCanvas.height = 0;
  showPreviewCanvas(false);
  imgInfoResult.textContent = "";
  downloadBtn.disabled      = true;
  downloadBtn.textContent   = "Descargar PNG";
  resultBlob = null;
  if (previewScaleTag) previewScaleTag.textContent = "";
}

function resetUI() {
  fileInput.value      = "";
  currentImage         = null;
  currentFileName      = "imagen";
  cancelRequested      = false;
  resultBlob           = null;
  dropZone.classList.remove("has-image");
  originalPreview.src  = "";
  sourceCtx.clearRect(0, 0, sourceCanvas.width, sourceCanvas.height);
  sourceCanvas.width   = 0;
  sourceCanvas.height  = 0;
  clearResultArea();
  imgInfoOrig.textContent = "";
  if (mMemEst) mMemEst.textContent = "—";
  if (mPad)    mPad.textContent    = "—";
  if (saveFSABtn) saveFSABtn.style.display = "none";
  processBtn.disabled  = !poolReady;
  clearBtn.textContent = "Limpiar";
  setProgress(0);
  setStatus(poolReady ? "Esperando imagen..." : "Iniciando worker pool...");
}

// ════════════════════════════════════════════════════════════════════════════
// §19  METRICS UI
// ════════════════════════════════════════════════════════════════════════════
function startMetricsUI(pool_, metrics_, cols, pad, total) {
  stopMetricsUI();
  metricsInterval = setInterval(() => {
    if (!mSpeed || !mWorkers || !mTiles || !mEta) return;
    const tps  = metrics_.tilesPerSec.toFixed(1);
    const eta  = metrics_.etaSec;
    const busy = pool_ ? pool_.busyCount : 0;
    const size = pool_ ? pool_.actualSize : 0;
    mSpeed.textContent   = `${tps} t/s`;
    mWorkers.textContent = `${busy}/${size} activos`;
    mTiles.textContent   = `${metrics_.done}/${total} (${metrics_.pct}%)`;
    mEta.textContent     = eta !== null ? `~${fmtSec(eta)}` : "—";
    setProgress(metrics_.pct);
  }, METRICS_HZ);
}

function stopMetricsUI() {
  if (metricsInterval) { clearInterval(metricsInterval); metricsInterval = null; }
}

// ════════════════════════════════════════════════════════════════════════════
// §20  UI UTILITIES
// ════════════════════════════════════════════════════════════════════════════
function setStatus(msg, isError = false) {
  statusText.textContent = msg;
  statusText.className   = "status" + (isError ? " status-error" : "");
}

function setProgress(pct) {
  progressFill.style.width  = pct + "%";
  progressBar.style.opacity = (pct > 0 && pct <= 100) ? "1" : "0";
}

function setProcessing(active) {
  isProcessing         = active;
  processBtn.disabled  = active || !poolReady || !currentImage;
  downloadBtn.disabled = active || !resultBlob;
  clearBtn.textContent = active ? "Cancelar" : "Limpiar";
  if (saveFSABtn) saveFSABtn.disabled = active;
}

function setModelDot(state) {
  const dot = document.getElementById("modelDot");
  if (!dot) return;
  dot.className = "model-dot model-dot--" + state;
}

function showPreviewCanvas(visible) {
  const ph = document.getElementById("resultPlaceholder");
  previewCanvas.style.display = visible ? "block" : "none";
  if (ph) ph.style.display = visible ? "none" : "flex";
}

function updateFSAButton(srcW, srcH) {
  if (!saveFSABtn) return;
  const hasFSA = "showSaveFilePicker" in window;
  saveFSABtn.style.display = hasFSA ? "inline-flex" : "none";
  const large = srcW > FSA_THRESHOLD || srcH > FSA_THRESHOLD;
  saveFSABtn.classList.toggle("btn-fsa-recommended", large);
}

function estimatePNGMB(srcW, srcH) {
  return Math.round((srcW * SCALE) * (srcH * SCALE) * 4 / 4 / (1024 * 1024));
}

function fmtSec(s) {
  if (s < 60)   return Math.ceil(s) + "s";
  if (s < 3600) return Math.floor(s / 60) + "m " + Math.ceil(s % 60) + "s";
  return Math.floor(s / 3600) + "h " + Math.ceil((s % 3600) / 60) + "m";
}
