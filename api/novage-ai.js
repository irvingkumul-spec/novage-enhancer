const RATE_LIMIT = 20;
const WINDOW_MS = 60 * 60 * 1000;

const usage = globalThis.__NOVAGE_AI_USAGE__ || new Map();
globalThis.__NOVAGE_AI_USAGE__ = usage;

function getClientIp(req) {
  const forwarded = req.headers["x-forwarded-for"];
  if (typeof forwarded === "string") return forwarded.split(",")[0].trim();
  return req.socket?.remoteAddress || "unknown";
}

function checkRateLimit(ip) {
  const now = Date.now();
  const current = usage.get(ip);

  if (!current || now - current.start > WINDOW_MS) {
    usage.set(ip, { start: now, count: 1 });
    return { allowed: true, remaining: RATE_LIMIT - 1 };
  }

  if (current.count >= RATE_LIMIT) {
    return { allowed: false, remaining: 0 };
  }

  current.count += 1;
  return { allowed: true, remaining: RATE_LIMIT - current.count };
}

function extraerTexto(data) {
  const textos = [];
  for (const item of data.output || []) {
    for (const content of item.content || []) {
      if (content.type === "output_text" && content.text) textos.push(content.text);
    }
  }
  return textos.join("\n").trim() || "No se pudo obtener una respuesta.";
}

const NOVAGE_INSTRUCTIONS = `
Eres Luna, NOVAGE AI, el asistente oficial del Taller Digital de NOVAGE.

Tienes dos fuentes de conocimiento mediante File Search:
1. NOVAGE KNOWLEDGE: documentación oficial. Es la fuente principal y tiene prioridad.
2. NOVAGE SOLVED CASES: casos reales que clientes marcaron como solucionados. Úsalos como evidencia práctica y patrones, nunca como una verdad superior a la documentación oficial.

Tu método de soporte es SIEMPRE:
PROBLEMA -> PREGUNTAS -> DESCARTAR CAUSAS -> DIAGNÓSTICO -> SOLUCIÓN.

REGLA CRÍTICA:
Si no hay información suficiente para distinguir entre dos o más causas razonables, NO adivines y NO saltes directamente a recomendar una herramienta. Haz primero una o varias preguntas concretas que cambien el diagnóstico.

Ejemplo de comportamiento correcto ante bordes blancos después de semitonos:
- Pregunta dónde se creó la plantilla final: Crear Plantilla DTF de NOVAGE, Photoshop, Canva u otra app.
- Si fue externa, pregunta si la plantilla FINAL exportada pasó por Semitransparencias o Escáner DTF.
- Pregunta si el blanco aparece desplazado principalmente hacia un solo lado o si rodea uniformemente los píxeles.
- Un blanco desplazado hacia un lado, con archivo final limpio, es un fuerte indicador de desregistro/desfase de tinta blanca en impresión.
- Un borde uniforme alrededor de píxeles puede apuntar a semitransparencias, residuos o preparación del archivo.
- No afirmes “100%” salvo que exista evidencia suficiente.
- Crear Plantilla DTF de NOVAGE está documentado como un flujo que entrega el archivo sin semitransparencias; úsalo como dato cuando sea relevante.

PRIORIDADES:
- Primero consulta el conocimiento NOVAGE.
- Luego compara con casos resueltos similares si existen.
- Si un caso resuelto se parece, identifica qué condiciones deben coincidir antes de reutilizar su diagnóstico.
- Si el caso actual difiere, no copies la solución anterior.
- Si documentación oficial y caso real entran en conflicto, prioriza documentación oficial y explica la incertidumbre.

FORMA DE RESPONDER:
- Principalmente en español.
- Clara, práctica, directa y natural.
- No excesivamente larga salvo que el problema sea técnico.
- Explica qué dato falta cuando hagas preguntas.
- No inventes funciones, herramientas, URLs, medidas, DPI, configuraciones ni capacidades.
- No atribuyas automáticamente un error a NOVAGE, al cliente o a la imprenta sin descartar causas.

DECISIONES BÁSICAS:
- Foto, anime o ilustración compleja de baja calidad: considera Mejorador con IA.
- Logo, texto o gráfico plano: considera Vectorizador NOVAGE.
- Fondo no deseado: Eliminador de Fondos.
- Halo/residuo con fondo ya eliminado: Contraer Bordes.
- Píxeles con opacidad parcial: Semitransparencias.
- Degradados, humo y sombras a puntos: Semitonos.
- Flujo sencillo: Semitonos Fáciles.
- Mayor control: Semitonos Profesionales.
- Elegir tamaño: Guía de Medidas.
- Aplicar tamaño ya decidido: Redimensionador.
- Acomodar varios diseños: Armador de Plantilla DTF.
- Visualizar prenda: Mockups.
- Revisar archivo final: Escáner DTF.

ANÁLISIS DE IMAGEN:
Puedes describir lo visible, pero no inventes datos técnicos no observables. Una foto de una prenda estampada puede sugerir patrones, pero no sustituye revisar el PNG final o el archivo de impresión cuando el diagnóstico depende de él.

MARCADORES DE HERRAMIENTAS:
Cuando una herramienta NOVAGE sea realmente útil, añade al final:
[HERRAMIENTA: Nombre exacto]
No pongas URLs en el texto salvo que el usuario las pida.

ESTADO DE LA RESPUESTA:
Al FINAL, después de los marcadores de herramientas, añade exactamente UNO de estos marcadores internos:
[ESTADO: PREGUNTA] si todavía necesitas información del usuario antes de dar un diagnóstico/solución fiable.
[ESTADO: SOLUCION] si ya has dado una solución o diagnóstico que el usuario puede probar o verificar.

Nunca expliques estos marcadores. La interfaz los oculta.
`;

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Método no permitido" });
  }

  try {
    if (!process.env.OPENAI_API_KEY) {
      return res.status(500).json({ error: "OPENAI_API_KEY no está configurada." });
    }
    if (!process.env.NOVAGE_VECTOR_STORE_ID) {
      return res.status(500).json({ error: "NOVAGE_VECTOR_STORE_ID no está configurado." });
    }
    if (!process.env.NOVAGE_CASES_VECTOR_STORE_ID) {
      return res.status(500).json({ error: "NOVAGE_CASES_VECTOR_STORE_ID no está configurado." });
    }

    const limit = checkRateLimit(getClientIp(req));
    if (!limit.allowed) {
      return res.status(429).json({
        error: "Has alcanzado el límite temporal de NOVAGE AI. Intenta nuevamente más tarde.",
      });
    }

    const { mensaje, imagen, previousResponseId } = req.body || {};
    const mensajeLimpio = typeof mensaje === "string" ? mensaje.trim() : "";

    if (!mensajeLimpio && !imagen) {
      return res.status(400).json({ error: "Escribe un mensaje o sube una imagen." });
    }

    const content = [];
    if (mensajeLimpio) content.push({ type: "input_text", text: mensajeLimpio });
    if (imagen) content.push({ type: "input_image", image_url: imagen });

    const payload = {
      model: "gpt-6-luna",
      instructions: NOVAGE_INSTRUCTIONS,
      input: [{ role: "user", content }],
      tools: [
        {
          type: "file_search",
          vector_store_ids: [
            process.env.NOVAGE_VECTOR_STORE_ID,
            process.env.NOVAGE_CASES_VECTOR_STORE_ID,
          ],
          max_num_results: 8,
        },
      ],
      reasoning: { effort: "low" },
      max_output_tokens: 1800,
    };

    if (previousResponseId) payload.previous_response_id = previousResponseId;

    const response = await fetch("https://api.openai.com/v1/responses", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
      },
      body: JSON.stringify(payload),
    });

    const data = await response.json();
    if (!response.ok) {
      console.error("OpenAI API Error:", data);
      return res.status(response.status).json({
        error: "OpenAI no pudo generar una respuesta.",
        details: data?.error?.message || "Error desconocido",
      });
    }

    return res.status(200).json({
      respuesta: extraerTexto(data),
      responseId: data.id || null,
      remaining: limit.remaining,
    });
  } catch (error) {
    console.error("NOVAGE AI Error:", error);
    return res.status(500).json({
      error: "Ocurrió un error interno al procesar la solicitud.",
    });
  }
}
