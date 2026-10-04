const RATE_LIMIT = 20;
const WINDOW_MS = 60 * 60 * 1000;

const usage =
  globalThis.__NOVAGE_AI_USAGE__ || new Map();

globalThis.__NOVAGE_AI_USAGE__ = usage;

function getClientIp(req) {
  const forwarded =
    req.headers["x-forwarded-for"];

  if (typeof forwarded === "string") {
    return forwarded
      .split(",")[0]
      .trim();
  }

  return (
    req.socket?.remoteAddress ||
    "unknown"
  );
}

function checkRateLimit(ip) {
  const now = Date.now();

  const current =
    usage.get(ip);

  if (
    !current ||
    now - current.start >
      WINDOW_MS
  ) {
    usage.set(ip, {
      start: now,
      count: 1,
    });

    return {
      allowed: true,
      remaining:
        RATE_LIMIT - 1,
    };
  }

  if (
    current.count >=
    RATE_LIMIT
  ) {
    return {
      allowed: false,
      remaining: 0,
    };
  }

  current.count += 1;

  return {
    allowed: true,
    remaining:
      RATE_LIMIT -
      current.count,
  };
}

function extraerTexto(data) {
  return (
    data.output
      ?.flatMap(
        (item) =>
          item.content || []
      )
      ?.find(
        (item) =>
          item.type ===
          "output_text"
      )
      ?.text ||
    "No se pudo obtener una respuesta."
  );
}

const NOVAGE_INSTRUCTIONS = `
Eres NOVAGE AI, el asistente oficial del Taller Digital de NOVAGE.

Tu función principal es ayudar a los usuarios utilizando prioritariamente la base de conocimiento oficial de NOVAGE.

IMPORTANTE:
- Antes de responder dudas relacionadas con NOVAGE, DTF, herramientas, errores, preparación de archivos, medidas, semitonos, vectorización, fondos, mockups, pedrería o cualquier función del Taller, utiliza File Search para consultar la base de conocimiento NOVAGE.
- Prioriza la información encontrada en la base de conocimiento sobre conocimiento genérico.
- No inventes características de NOVAGE.
- No inventes herramientas.
- No inventes enlaces.
- No inventes configuraciones.
- Si la base de conocimiento no contiene suficiente información para afirmar algo, dilo claramente.

FORMA DE RESPONDER:
- Responde principalmente en español.
- Sé claro, práctico y directo.
- Evita respuestas genéricas.
- Explica brevemente el problema.
- Da una solución concreta.
- Recomienda la herramienta NOVAGE adecuada cuando corresponda.
- Si hay varias opciones, explica cuál conviene más.
- No recomiendes herramientas solo por rellenar la respuesta.

RAZONAMIENTO DENTRO DE NOVAGE:
- Fotografía, anime o ilustración compleja de baja calidad: considera Mejorador con IA.
- Logo, texto o gráfico plano con pocos colores: considera Vectorizador NOVAGE.
- Fondo no deseado: Eliminador de Fondos.
- Halo o borde blanco: Contraer Bordes.
- Píxeles con opacidad parcial: Semitransparencias.
- Degradados, humo, sombras o transparencias que necesitan convertirse para DTF: Semitonos.
- Usuario principiante que quiere rapidez: Semitonos Fáciles.
- Usuario que necesita mayor control: Semitonos Profesionales.
- No sabe qué tamaño utilizar: Guía de Medidas.
- Ya sabe el tamaño y necesita aplicarlo: Redimensionador.
- Quiere acomodar varios diseños: Armador de Plantilla DTF.
- Quiere visualizar una prenda: Mockups.
- Quiere saber cuánto cobrar: Calculadora de Precios.
- Quiere revisar un archivo: Escáner DTF.

ANÁLISIS DE IMÁGENES:
Cuando el usuario suba una imagen:
- Analízala visualmente.
- Busca pixelación aparente.
- Revisa bordes blancos.
- Revisa halos.
- Revisa fondos.
- Revisa líneas demasiado finas.
- Revisa detalles muy pequeños.
- Revisa degradados, humo o zonas transparentes visibles.
- Determina si parece más adecuado mejorar o vectorizar.
- No inventes DPI.
- No inventes medidas físicas.
- No inventes resolución real si no está disponible.

FORMATO PARA HERRAMIENTAS:
Cuando recomiendes una herramienta disponible en NOVAGE, añade al final:

[HERRAMIENTA: Nombre exacto]

Ejemplos:
[HERRAMIENTA: Vectorizador NOVAGE]
[HERRAMIENTA: Eliminador de Fondos]
[HERRAMIENTA: Contraer Bordes]

El frontend de NOVAGE convertirá estos marcadores en botones y tarjetas.

No escribas URLs directamente en la respuesta salvo que el usuario las pida.
`;

export default async function handler(
  req,
  res
) {
  if (req.method !== "POST") {
    return res
      .status(405)
      .json({
        error:
          "Método no permitido",
      });
  }

  try {
    if (
      !process.env.OPENAI_API_KEY
    ) {
      return res
        .status(500)
        .json({
          error:
            "OPENAI_API_KEY no está configurada.",
        });
    }

    if (
      !process.env
        .NOVAGE_VECTOR_STORE_ID
    ) {
      return res
        .status(500)
        .json({
          error:
            "NOVAGE_VECTOR_STORE_ID no está configurado.",
        });
    }

    const ip =
      getClientIp(req);

    const limit =
      checkRateLimit(ip);

    if (!limit.allowed) {
      return res
        .status(429)
        .json({
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
      (!mensaje ||
        !mensaje.trim()) &&
      !imagen
    ) {
      return res
        .status(400)
        .json({
          error:
            "Escribe un mensaje o sube una imagen.",
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

      instructions:
        NOVAGE_INSTRUCTIONS,

      input: [
        {
          role: "user",
          content,
        },
      ],

      tools: [
        {
          type: "file_search",

          vector_store_ids: [
            process.env
              .NOVAGE_VECTOR_STORE_ID,
          ],

          max_num_results: 6,
        },
      ],

      reasoning: {
        effort: "low",
      },

      max_output_tokens: 1800,
    };

    if (previousResponseId) {
      payload.previous_response_id =
        previousResponseId;
    }

    const response =
      await fetch(
        "https://api.openai.com/v1/responses",
        {
          method: "POST",

          headers: {
            "Content-Type":
              "application/json",

            Authorization:
              `Bearer ${process.env.OPENAI_API_KEY}`,
          },

          body:
            JSON.stringify(
              payload
            ),
        }
      );

    const data =
      await response.json();

    if (!response.ok) {
      console.error(
        "OpenAI API Error:",
        data
      );

      return res
        .status(
          response.status
        )
        .json({
          error:
            "OpenAI no pudo generar una respuesta.",

          details:
            data?.error
              ?.message ||
            "Error desconocido",
        });
    }

    return res
      .status(200)
      .json({
        respuesta:
          extraerTexto(data),

        responseId:
          data.id,

        remaining:
          limit.remaining,
      });
  } catch (error) {
    console.error(
      "NOVAGE AI Error:",
      error
    );

    return res
      .status(500)
      .json({
        error:
          "Ocurrió un error interno.",
      });
  }
}
