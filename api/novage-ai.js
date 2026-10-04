const RATE_LIMIT = 20;
const WINDOW_MS = 60 * 60 * 1000;

const usage = globalThis.__NOVAGE_AI_USAGE__ || new Map();
globalThis.__NOVAGE_AI_USAGE__ = usage;

function getClientIp(req) {
  const forwarded = req.headers["x-forwarded-for"];

  if (typeof forwarded === "string") {
    return forwarded.split(",")[0].trim();
  }

  return req.socket?.remoteAddress || "unknown";
}

function checkRateLimit(ip) {
  const now = Date.now();

  const current = usage.get(ip);

  if (!current || now - current.start > WINDOW_MS) {
    usage.set(ip, {
      start: now,
      count: 1,
    });

    return {
      allowed: true,
      remaining: RATE_LIMIT - 1,
    };
  }

  if (current.count >= RATE_LIMIT) {
    return {
      allowed: false,
      remaining: 0,
    };
  }

  current.count += 1;

  return {
    allowed: true,
    remaining: RATE_LIMIT - current.count,
  };
}

function extraerTexto(data) {
  return (
    data.output
      ?.flatMap((item) => item.content || [])
      ?.find((item) => item.type === "output_text")
      ?.text ||
    "No se pudo obtener una respuesta."
  );
}

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({
      error: "Método no permitido",
    });
  }

  try {
    const ip = getClientIp(req);

    const limit = checkRateLimit(ip);

    if (!limit.allowed) {
      return res.status(429).json({
        error:
          "Has alcanzado el límite temporal de NOVAGE AI. Intenta nuevamente más tarde.",
      });
    }

    const {
      mensaje,
      imagen,
      previousResponseId,
    } = req.body || {};

    if (
      (!mensaje || !mensaje.trim()) &&
      !imagen
    ) {
      return res.status(400).json({
        error: "Escribe un mensaje o sube una imagen.",
      });
    }

    const content = [];

    if (mensaje?.trim()) {
      content.push({
        type: "input_text",
        text: mensaje.trim(),
      });
    }

    if (imagen) {
      content.push({
        type: "input_image",
        image_url: imagen,
      });
    }

    const payload = {
      model: "gpt-6-luna",

      instructions: `
Eres NOVAGE AI, el asistente especializado de NOVAGE.

Tu especialidad es:
- Impresión DTF.
- Preparación de diseños.
- Calidad y resolución.
- Eliminación de fondos.
- Vectorización.
- Semitonos.
- Semitransparencias.
- Mockups.
- Medidas de impresión.
- Preparación de archivos PNG.
- Revisión visual de diseños para impresión.

Cuando el usuario suba una imagen:
- Analízala visualmente.
- Describe problemas relevantes para DTF.
- Señala bordes blancos, halos, fondos, pixelación,
  baja resolución aparente, semitransparencias,
  zonas demasiado finas o detalles que puedan dar problemas.
- No inventes información técnica que no puedas inferir visualmente.
- Si no puedes determinar algo solo mirando la imagen,
  indícalo claramente.

Herramientas NOVAGE disponibles:
- Mejorador de imágenes.
- Eliminador de fondos.
- Vectorizador.
- Semitonos Pro.
- Semitonos Fáciles.
- Mockups.
- Armador DTF.

Cuando una herramienta de NOVAGE sea útil,
recomiéndala de forma natural.

Responde principalmente en español.

Sé claro, directo y práctico.
Evita respuestas innecesariamente largas.
      `,

      input: [
        {
          role: "user",
          content,
        },
      ],

      reasoning: {
        effort: "low",
      },

      max_output_tokens: 1200,
    };

    if (previousResponseId) {
      payload.previous_response_id =
        previousResponseId;
    }

    const response = await fetch(
      "https://api.openai.com/v1/responses",
      {
        method: "POST",

        headers: {
          "Content-Type": "application/json",

          Authorization:
            `Bearer ${process.env.OPENAI_API_KEY}`,
        },

        body: JSON.stringify(payload),
      }
    );

    const data = await response.json();

    if (!response.ok) {
      console.error("OpenAI API Error:", data);

      return res
        .status(response.status)
        .json({
          error:
            "OpenAI no pudo generar una respuesta.",

          details:
            data?.error?.message ||
            "Error desconocido",
        });
    }

    return res.status(200).json({
      respuesta: extraerTexto(data),

      responseId: data.id,

      remaining: limit.remaining,
    });
  } catch (error) {
    console.error("NOVAGE AI Error:", error);

    return res.status(500).json({
      error: "Ocurrió un error interno.",
    });
  }
}
