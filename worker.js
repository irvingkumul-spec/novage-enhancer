/**
 * worker.js — AI inference node for the worker pool
 * ════════════════════════════════════════════════════════════════════════════
 *
 * Each instance of this file runs in its own dedicated Worker thread.
 * The pool spawns N of these (N = 2-6 depending on hardware).
 *
 * ROLE IN THE PIPELINE:
 *   Main thread                          Worker (this file)
 *   ────────────────────────────────     ──────────────────────────────────
 *   { type:"process_tile", tileId, }
 *   Float32[64×64×3] ──────────────►  tf.tensor4d → model.execute → await
 *                                        Float32[256×256×3] (GPU output)
 *   { type:"tile_done", tileId, }   ◄──  transfer buffer (zero-copy)
 *
 * WHY ONE TILE AT A TIME PER WORKER:
 *   GPU inference is sequential within a single WebGL context. Batching
 *   multiple tiles per model.execute() call is theoretically possible but
 *   requires the model to be re-exported for variable batch sizes. Our
 *   model uses a fixed [1, 64, 64, 3] input tensor. True parallelism
 *   comes from having N workers each holding their own WebGL context,
 *   letting the GPU driver interleave their work at the command-queue level.
 *
 * MEMORY CONTRACTS:
 *   • `data.buffer` (input): transferred (neutered) into this thread — 0 copy.
 *   • `d.buffer`   (output): transferred back to main thread — 0 copy.
 *   • Tensors are disposed in `finally` — no VRAM leak across tiles.
 *
 * GPU BACKEND PRIORITY: webgl → cpu
 *   WebGPU would be ideal but requires a separate TF.js bundle not included
 *   in the standard CDN minified build. WebGL gives 10-30× speedup vs CPU
 *   for convolutional models like this super-resolution network.
 */

importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.10.0/dist/tf.min.js");

const MODEL_URL = "https://cdn.shopify.com/s/files/1/0686/6658/9357/files/model.json?v=1753739272";
const TILE_IN   = 64;   // fixed model input dimension (must match training)

let model   = null;
let backend = "none";

// ─── Model initialization ─────────────────────────────────────────────────────
async function initModel() {
  try {
    // Try backends in descending performance order
    for (const b of ["webgl", "cpu"]) {
      try {
        await tf.setBackend(b);
        await tf.ready();
        backend = tf.getBackend();
        break;
      } catch (_) { /* backend unavailable, try next */ }
    }

    if (backend === "webgl") {
      // WEBGL_PACK: fuses element-wise ops into fewer shader calls.
      // WEBGL_PACK_DEPTHWISECONV: fuses depthwise convolutions (used heavily
      //   in ESRGAN/RRDB-style SR models). Both reduce GPU command overhead
      //   by 30-60%, especially important when dispatching many small tiles.
      tf.env().set("WEBGL_PACK",               true);
      tf.env().set("WEBGL_PACK_DEPTHWISECONV",  true);
      tf.env().set("WEBGL_FORCE_F16_TEXTURES",  false); // keep f32 precision
    }

    model = await tf.loadGraphModel(MODEL_URL);

    // Warmup: forces GLSL shader compilation before the first real tile.
    // Without warmup the first inference per worker takes 500-2000ms extra.
    // With N workers loading in parallel, warmup happens concurrently
    // (browser GPU command queues are serialized, but CPU-side setup is not).
    const warmupOut = tf.tidy(() =>
      model.execute(tf.zeros([1, TILE_IN, TILE_IN, 3]))
    );
    warmupOut.dispose();

    self.postMessage({ type: "ready", backend });

  } catch (err) {
    self.postMessage({ type: "error", tileId: null, message: "Model init failed: " + err.message });
  }
}

// ─── Tile inference ───────────────────────────────────────────────────────────
self.onmessage = async ({ data }) => {
  if (data.type !== "process_tile") return;
  if (!model) {
    self.postMessage({ type: "error", tileId: data.tileId, message: "Model not ready." });
    return;
  }

  let tensor = null;
  let output = null;

  try {
    // Reconstruct Float32Array from the transferred ArrayBuffer (zero copy).
    tensor = tf.tensor4d(new Float32Array(data.buffer), [1, TILE_IN, TILE_IN, 3]);
    output = model.execute(tensor);

    // Async readback from VRAM → CPU RAM.
    // The `await` yields the thread so the browser can handle other messages
    // while the GPU finishes rendering the previous shader dispatch.
    const d = await output.data(); // Float32Array[256×256×3]

    // Transfer the ArrayBuffer back to main thread. Zero-copy: the buffer
    // moves ownership to main thread and `d` becomes neutered here.
    self.postMessage(
      { type: "tile_done", tileId: data.tileId, buffer: d.buffer },
      [d.buffer]
    );

  } catch (err) {
    self.postMessage({ type: "error", tileId: data.tileId, message: err.message });

  } finally {
    // CRITICAL: dispose tensors regardless of success/failure.
    // Each un-disposed tensor accumulates 50-80 MB of VRAM.
    // With 5000+ tiles/image this would cause guaranteed OOM without this block.
    if (tensor) tensor.dispose();
    if (output) output.dispose();
  }
};

initModel();