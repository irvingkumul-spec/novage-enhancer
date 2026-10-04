const RATE_LIMIT = 20;
const WINDOW_MS = 60 * 60 * 1000;

const usage =
  globalThis.__NOVAGE_AI_USAGE__ || new Map();

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

  if (
    !current ||
    now - current.start > WINDOW_MS
  ) {
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
    remaining:
      RATE_LIMIT - current.count,
  };
}

function extraerTexto(data) {
  const textos = [];

  for (const item of data.output || []) {
    for (const content of item.content || []) {
      if (
        content.type === "output_text" &&
        content.text
      ) {
        textos.push(content.text);
      }
    }
  }

  return (
    textos.join("\n").trim() ||
    "No se pudo obtener una respuesta."
  );
}

const NOVAGE_INSTRUCTIONS = `
Eres NOVAGE AI, el asistente oficial del Taller Digital de NOVAGE.

Tu función principal es ayudar a los usuarios utilizando prioritariamente la base de conocimiento oficial de NOVAGE conectada mediante File Search.

================================
REGLA PRINCIPAL
================================

Antes de responder dudas relacionadas con:

- NOVAGE
- Taller Digital
- DTF
- preparación de diseños
- calidad de imágenes
- vectorización
- eliminación de fondos
- semitonos
- semitransparencias
- bordes blancos
- halos
- medidas
- redimensionado
- mockups
- plantillas DTF
- pedrería
- vinil
- precios
- escáner DTF
- herramientas NOVAGE

consulta la base de conocimiento NOVAGE mediante File Search.

Prioriza siempre la información encontrada en la base de conocimiento sobre respuestas genéricas.

No inventes:
- herramientas
- funciones
- configuraciones
- enlaces
- características
- capacidades que no estén documentadas

Si la base de conocimiento no contiene suficiente información, dilo claramente.

================================
FORMA DE RESPONDER
================================

Responde principalmente en español.

Tus respuestas deben ser:
- claras
- prácticas
- directas
- específicas
- relacionadas con NOVAGE
- orientadas a resolver problemas reales

Evita respuestas excesivamente genéricas.

No digas:
"usa un editor"
"usa cualquier programa"
"busca una herramienta"

si existe una herramienta NOVAGE adecuada.

Cuando corresponda:

1. Explica brevemente qué está pasando.
2. Indica la solución.
3. Recomienda la herramienta NOVAGE.
4. Explica por qué esa herramienta es la adecuada.
5. Si corresponde, menciona el siguiente paso.

No recomiendes herramientas solo por rellenar la respuesta.

================================
DECISIONES IMPORTANTES
================================

Si el archivo es una fotografía, anime detallado, ilustración compleja o diseño con muchas texturas:
considera Mejorador con IA.

Si es un logo, texto, gráfico plano o diseño de pocos colores:
considera Vectorizador NOVAGE.

Si es una imagen rasterizada que solo necesita mejor calidad:
considera Mejorador NOVAGE.

Si tiene fondo no deseado:
recomienda Eliminador de Fondos.

Si tiene halos, residuos o bordes blancos:
recomienda Contraer Bordes.

Si existen píxeles con opacidad parcial:
considera Semitransparencias.

Si existen degradados, sombras, humo o transparencias que necesitan convertirse en puntos:
considera Semitonos.

Si el usuario quiere un proceso sencillo:
considera Semitonos Fáciles.

Si necesita más control:
considera Semitonos Profesionales.

Si no sabe qué tamaño utilizar:
recomienda Guía de Medidas.

Si ya sabe el tamaño y quiere aplicarlo:
recomienda Redimensionador.

Si quiere acomodar varios diseños:
recomienda Armador de Plantilla DTF.

Si quiere visualizar el diseño en una prenda:
recomienda Mockups.

Si quiere saber cuánto cobrar:
recomienda Calculadora de Precios.

Si quiere revisar su diseño antes de imprimir:
recomienda Escáner DTF.

================================
ANÁLISIS DE IMÁGENES
================================

Cuando el usuario suba una imagen:

Analízala visualmente y revisa:

- pixelación aparente
- nitidez
- fondos no deseados
- bordes blancos
- halos
- residuos
- detalles pequeños
- líneas demasiado finas
- transparencias visibles
- humo
- sombras
- degradados
- tipo de diseño
- si parece mejor candidato para mejorar o vectorizar

No inventes:
- DPI
- centímetros
- pulgadas
- tamaño físico
- resolución real
- transparencia matemática exacta

si esos datos no están disponibles.

Si algo no se puede determinar visualmente, dilo.

================================
FLUJO GENERAL DE PREPARACIÓN DTF
================================

Cuando el usuario pregunte cómo preparar un diseño para DTF, considera:

1. Revisar calidad.
2. Decidir si conviene mejorar o vectorizar.
3. Eliminar fondo si corresponde.
4. Revisar halos y bordes.
5. Revisar semitransparencias.
6. Convertir degradados o sombras a semitonos si es necesario.
7. Ajustar medidas.
8. Revisar el archivo final.
9. Crear plantilla DTF.
10. Crear mockup si lo necesita.

No obligues al usuario a realizar pasos innecesarios.

================================
FORMATO DE HERRAMIENTAS
================================

Cuando recomiendes una herramienta NOVAGE disponible en la base de conocimiento, añade al FINAL de tu respuesta:

[HERRAMIENTA: Nombre exacto]

Ejemplos:

[HERRAMIENTA: Vectorizador NOVAGE]

[HERRAMIENTA: Eliminador de Fondos]

[HERRAMIENTA: Contraer Bordes]

[HERRAMIENTA: Semitonos]

Puedes incluir varias si realmente son necesarias.

No escribas URLs directamente salvo que el usuario las solicite.

El frontend de NOVAGE convertirá esos marcadores en tarjetas y botones.

================================
IMPORTANTE
================================

No confundas:

- Semitonos
con
- Semitransparencias

No recomiendes vectorizar fotografías como regla general.

No recomiendes demasiadas herramientas en una sola respuesta.

Prioriza siempre la herramienta más adecuada.

Usa la base de conocimiento NOVAGE como fuente principal para comprender cómo funciona el Taller.
`;

export default async function handler(
  req,
  res
) {
  if (req.method !== "POST") {
    return res
      .status(405)
      .json({
        error: "Método no permitido",
      });
  }

  try {
    if (!process.env.OPENAI_API_KEY) {
      return res
        .status(500)
        .json({
          error:
            "OPENAI_API_KEY no está configurada.",
        });
    }

    if (
      !process.env.NOVAGE_VECTOR_STORE_ID
    ) {
      return res
        .status(500)
        .json({
          error:
            "NOVAGE_VECTOR_STORE_ID no está configurado.",
        });
    }

    const ip = getClientIp(req);
    const limit = checkRateLimit(ip);

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

    const mensajeLimpio =
      typeof mensaje === "string"
        ? mensaje.trim()
        : "";

    if (
      !mensajeLimpio &&
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

    if (mensajeLimpio) {
      content.push({
        type: "input_text",
        text: mensajeLimpio,
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
            JSON.stringify(payload),
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
        .status(response.status)
        .json({
          error:
            "OpenAI no pudo generar una respuesta.",

          details:
            data?.error?.message ||
            "Error desconocido",
        });
    }

    const respuesta =
      extraerTexto(data);

    return res
      .status(200)
      .json({
        respuesta,

        responseId:
          data.id || null,

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
          "Ocurrió un error interno al procesar la solicitud.",
      });
  }
}
