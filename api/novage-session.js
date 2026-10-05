// NOVAGE AI • Sesión protegida • v5.1
// Emite una sesión corta SOLO después de validar Cloudflare Turnstile.
// Requiere en Vercel:
//   TURNSTILE_SECRET_KEY
//   NOVAGE_SESSION_SECRET

import crypto from "crypto";

const ALLOWED_ORIGINS = new Set([
  "https://novage.store",
  "https://www.novage.store"
]);

const ALLOWED_TURNSTILE_HOSTNAMES = new Set([
  "novage.store",
  "www.novage.store"
]);

const API_HOST = "tools.novage.store";
const SESSION_TTL_SECONDS = 45 * 60;

const SESSION_RATE_LIMIT = 20;
const SESSION_WINDOW_MS = 10 * 60 * 1000;

const usage = globalThis.__NOVAGE_SESSION_USAGE__ || new Map();
globalThis.__NOVAGE_SESSION_USAGE__ = usage;

function getOrigin(req) {
  return String(req.headers.origin || "").trim();
}

function getHost(req) {
  return String(req.headers["x-forwarded-host"] || req.headers.host || "")
    .split(",")[0]
    .trim()
    .toLowerCase();
}

function getClientIp(req) {
  const forwarded = req.headers["x-forwarded-for"];
  if (typeof forwarded === "string" && forwarded.trim()) {
    return forwarded.split(",")[0].trim();
  }
  return req.socket?.remoteAddress || "unknown";
}

function getUserAgent(req) {
  return String(req.headers["user-agent"] || "").slice(0, 600);
}

function checkRateLimit(ip) {
  const now = Date.now();
  const current = usage.get(ip);

  if (!current || now - current.start > SESSION_WINDOW_MS) {
    usage.set(ip, { start: now, count: 1 });
    return true;
  }

  if (current.count >= SESSION_RATE_LIMIT) return false;

  current.count += 1;
  return true;
}

function setCors(req, res) {
  const origin = getOrigin(req);

  if (ALLOWED_ORIGINS.has(origin)) {
    res.setHeader("Access-Control-Allow-Origin", origin);
    res.setHeader("Vary", "Origin");
  }

  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
  res.setHeader("Access-Control-Max-Age", "600");
}

function base64url(input) {
  return Buffer.from(input).toString("base64url");
}

function fingerprint(req) {
  const ip = getClientIp(req);
  const ua = getUserAgent(req);

  return crypto
    .createHash("sha256")
    .update(`${ip}|${ua}`)
    .digest("base64url");
}

function signPayload(payload, secret) {
  const body = base64url(JSON.stringify(payload));
  const signature = crypto
    .createHmac("sha256", secret)
    .update(body)
    .digest("base64url");

  return `${body}.${signature}`;
}

async function verifyTurnstile(token, remoteIp) {
  const body = new URLSearchParams();
  body.set("secret", process.env.TURNSTILE_SECRET_KEY || "");
  body.set("response", token);

  if (remoteIp && remoteIp !== "unknown") {
    body.set("remoteip", remoteIp);
  }

  const response = await fetch(
    "https://challenges.cloudflare.com/turnstile/v0/siteverify",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded"
      },
      body
    }
  );

  const data = await response.json();

  if (!response.ok || data?.success !== true) {
    const codes = Array.isArray(data?.["error-codes"])
      ? data["error-codes"].join(", ")
      : "verificación rechazada";

    throw new Error(`Turnstile rechazó la verificación: ${codes}`);
  }

  const hostname = String(data?.hostname || "").toLowerCase();

  if (!ALLOWED_TURNSTILE_HOSTNAMES.has(hostname)) {
    throw new Error("El token de Turnstile no pertenece a NOVAGE.");
  }

  return data;
}

export default async function handler(req, res) {
  setCors(req, res);

  const origin = getOrigin(req);
  const host = getHost(req);
  const secFetchSite = String(req.headers["sec-fetch-site"] || "").toLowerCase();

  if (!ALLOWED_ORIGINS.has(origin)) {
    return res.status(403).json({ error: "Origen no autorizado." });
  }

  if (host !== API_HOST) {
    return res.status(403).json({ error: "Host no autorizado." });
  }

  if (
    secFetchSite &&
    secFetchSite !== "same-site" &&
    secFetchSite !== "same-origin"
  ) {
    return res.status(403).json({
      error: "Contexto de navegación no autorizado."
    });
  }

  if (req.method === "OPTIONS") {
    return res.status(204).end();
  }

  if (req.method !== "POST") {
    return res.status(405).json({ error: "Método no permitido." });
  }

  if (!process.env.TURNSTILE_SECRET_KEY) {
    return res.status(500).json({
      error: "TURNSTILE_SECRET_KEY no está configurada."
    });
  }

  if (!process.env.NOVAGE_SESSION_SECRET) {
    return res.status(500).json({
      error: "NOVAGE_SESSION_SECRET no está configurada."
    });
  }

  const ip = getClientIp(req);

  if (!checkRateLimit(ip)) {
    return res.status(429).json({
      error: "Demasiados intentos de verificación. Intenta más tarde."
    });
  }

  const turnstileToken =
    typeof req.body?.turnstileToken === "string"
      ? req.body.turnstileToken.trim()
      : "";

  if (!turnstileToken || turnstileToken.length > 4096) {
    return res.status(400).json({
      error: "Falta la verificación de seguridad."
    });
  }

  try {
    await verifyTurnstile(turnstileToken, ip);

    const now = Math.floor(Date.now() / 1000);

    const payload = {
      aud: "novage-ai",
      iat: now,
      exp: now + SESSION_TTL_SECONDS,
      fp: fingerprint(req),
      jti: crypto.randomUUID()
    };

    const sessionToken = signPayload(
      payload,
      process.env.NOVAGE_SESSION_SECRET
    );

    res.setHeader("Cache-Control", "no-store, max-age=0");
    res.setHeader("Pragma", "no-cache");

    return res.status(200).json({
      sessionToken,
      expiresIn: SESSION_TTL_SECONDS
    });
  } catch (error) {
    console.error("NOVAGE session error:", error);

    return res.status(403).json({
      error: "No se pudo verificar el acceso a NOVAGE AI.",
      details: error?.message || "Verificación rechazada."
    });
  }
}
