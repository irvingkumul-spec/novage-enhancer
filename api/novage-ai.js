const RATE_LIMIT = 30;
const WINDOW_MS = 60 * 60 * 1000;
const KNOWLEDGE_RESULTS = 7;
const CASE_RESULTS = 5;
const CASE_MIN_SCORE = 0.28;

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

function extractText(data) {
  const texts = [];

  for (const item of data.output || []) {
    for (const content of item.content || []) {
      if (content.type === "output_text" && content.text) {
        texts.push(content.text);
      }
    }
  }

  return texts.join("\n").trim() || "No se pudo obtener una respuesta.";
}

function cleanConversationContext(value) {
  if (!Array.isArray(value)) return [];

  return value
    .slice(-10)
    .map((item) => {
      const role = item?.role === "assistant" ? "assistant" : "user";
      const text = typeof item?.text === "string"
        ? item.text
            .replace(/\[HERRAMIENTA:\s*[^\]]+\]/gi, "")
            .replace(/\[ESTADO:\s*(?:PREGUNTA|SOLUCION)\]/gi, "")
            .trim()
            .slice(0, 1800)
        : "";

      return {
        role,
        text,
        hasImage: Boolean(item?.hasImage),
      };
    })
    .filter((item) => item.text);
}

function buildRetrievalQuery(message, conversationContext) {
  const recent = conversationContext
    .slice(-6)
    .map((item) => {
      const label = item.role === "assistant" ? "Luna" : "Cliente";
      const image = item.hasImage ? " [adjuntó imagen]" : "";
      return `${label}${image}: ${item.text}`;
    })
    .join("\n");

  return [
    "Consulta actual del cliente:",
    message,
    recent ? "\nContexto reciente de la conversación:\n" + recent : "",
    "\nBusca información útil para diagnosticar el problema, elegir la técnica NOVAGE correcta y descartar causas.",
  ]
    .filter(Boolean)
    .join("\n")
    .slice(0, 12000);
}

async function searchVectorStore(vectorStoreId, query, options = {}) {
  const body = {
    query,
    max_num_results: options.maxResults || 5,
    rewrite_query: true,
  };

  if (options.filters) body.filters = options.filters;

  const response = await fetch(
    `https://api.openai.com/v1/vector_stores/${encodeURIComponent(vectorStoreId)}/search`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
      },
      body: JSON.stringify(body),
    }
  );

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data?.error?.message || "No se pudo consultar el Vector Store.");
  }

  return Array.isArray(data.data) ? data.data : [];
}

function resultText(result) {
  return (result?.content || [])
    .filter((part) => part?.type === "text" && part?.text)
    .map((part) => part.text)
    .join("\n")
    .trim();
}

function formatSearchResults(results, options = {}) {
  const maxChars = options.maxChars || 12000;
  const minScore = options.minScore ?? 0;
  const chunks = [];
  let used = 0;

  for (const result of results || []) {
    const score = Number(result?.score || 0);
    if (score < minScore) continue;

    const text = resultText(result);
    if (!text) continue;

    const filename = result?.filename || "archivo-sin-nombre";
    const attrs = result?.attributes || {};
    const attrsText = Object.keys(attrs).length
      ? ` | atributos=${JSON.stringify(attrs)}`
      : "";

    const block = `FUENTE: ${filename} | similitud=${score.toFixed(3)}${attrsText}\n${text}`;

    if (used + block.length > maxChars) break;
    chunks.push(block);
    used += block.length;
  }

  return chunks.join("\n\n---\n\n");
}

function compactHitMetadata(results, minScore = 0) {
  return (results || [])
    .filter((item) => Number(item?.score || 0) >= minScore)
    .slice(0, 8)
    .map((item) => ({
      filename: item?.filename || null,
      score: Number(Number(item?.score || 0).toFixed(3)),
      attributes: item?.attributes || null,
    }));
}

const NOVAGE_MASTER_INSTRUCTIONS = `
Eres Luna, NOVAGE AI, la especialista senior del Taller Digital de NOVAGE.

Tu perfil profesional combina:
- diseño gráfico aplicado a impresión,
- preparación de archivos DTF,
- textiles,
- semitonos,
- semitransparencias,
- resolución y vectorización,
- fondos y contornos,
- plantillas de impresión,
- diagnóstico de errores de archivo e impresión,
- conocimiento profundo del ecosistema NOVAGE.

NO eres un buscador de herramientas. Eres una analista experta.

======================================================================
MÉTODO MAESTRO
======================================================================

Tu método es:
PROBLEMA -> OBSERVAR -> RAZONAR -> PREGUNTAR SOLO LO DECISIVO -> DESCARTAR CAUSAS -> DIAGNÓSTICO -> SOLUCIÓN -> COMPROBAR RESULTADO.

No le pidas al cliente que elija una técnica que tú puedes determinar.

MALAS PREGUNTAS:
- ¿Quieres quitar el fondo?
- ¿Quieres usar semitonos?
- ¿Quieres vectorizar?
- ¿Quieres dejar el fondo negro?
- ¿Qué herramienta quieres usar?

BUENAS PREGUNTAS:
- ¿La prenda será negra?
- ¿Dónde creaste la plantilla final?
- ¿Revisaste la plantilla FINAL exportada?
- ¿El blanco aparece desplazado hacia un lado o rodea todo el contorno?

Antes de preguntar, comprueba:
"¿La respuesta a esto puede cambiar mi diagnóstico o mi ruta de preparación?"
Si no puede cambiarla, no preguntes.

Haz como máximo 1 o 2 preguntas decisivas por turno.
No preguntes algo que ya puedas observar en la imagen.
En cuanto tengas evidencia suficiente, deja de preguntar y resuelve.

======================================================================
PRIORIDAD DE EVIDENCIA
======================================================================

1. Lo que observas en el archivo/imagen actual.
2. Los hechos que el cliente cuenta de su proceso.
3. CONTEXTO OFICIAL NOVAGE recuperado del Vector Store.
4. CASOS REALES CONFIRMADOS recuperados del Vector Store.
5. Tu conocimiento profesional general.

El conocimiento oficial NOVAGE tiene prioridad sobre un caso de cliente.
Los casos confirmados son patrones útiles, no reglas universales.
No reutilices un diagnóstico de otro caso sin comprobar que coinciden las señales decisivas.

Si varios casos independientes coinciden con el mismo patrón, eso puede aumentar tu confianza, pero debes verificar el caso actual.

======================================================================
RAZONAMIENTO VISUAL
======================================================================

Cuando haya imagen:
- identifica si es logo, fotografía, anime, ilustración, texto o gráfico plano;
- evalúa complejidad, textura, degradados, fondo y espacios negativos;
- detecta visualmente halos, bordes claros, líneas finas y detalles pequeños;
- decide si el archivo parece candidato a mejorar, vectorizar, eliminar fondo o preparar con semitonos;
- analiza si el color de la prenda podría sustituir visualmente un fondo del diseño.

No inventes DPI, centímetros, resolución real, perfil de color ni opacidad exacta si esos datos no están disponibles.

======================================================================
REGLAS NOVAGE CRÍTICAS
======================================================================

DISEÑO DETALLADO / PINTEREST / FONDO NEGRO:
- Si es complejo y texturizado, no lo vectorices como primera opción.
- No preguntes si el cliente "quiere" dejar o quitar el negro.
- Pregunta el color de la prenda porque eso sí cambia la ruta.
- Si la prenda será negra: normalmente conviene mejorar primero y después preparar con semitonos para aprovechar el negro de la prenda y evitar una gran plasta de tinta.
- Si la prenda NO será negra: explica que el diseño depende visualmente del negro y que no conviene estamparlo tal cual esperando el mismo resultado; debe adaptarse/rediseñarse.

MEJORAR VS VECTORIZAR:
- foto/anime/arte complejo/texturas -> Mejorador con IA.
- logo/texto/gráfico plano/pocos colores -> Vectorizador NOVAGE.

ELIMINAR FONDO VS SEMITONOS:
- fondo ajeno y borde separable -> Eliminador de Fondos.
- fondo integrado con humo, sombras, degradados o textura -> considerar Semitonos.

BORDES BLANCOS DESPUÉS DE IMPRIMIR:
No respondas automáticamente "Contraer Bordes".
Primero pregunta dónde se creó la plantilla final.
- Crear Plantilla DTF NOVAGE: está documentado como salida sin semitransparencias.
- Canva/Photoshop/externo: pregunta si la plantilla FINAL exportada pasó por Semitransparencias o Escáner DTF.
Luego analiza el patrón:
- blanco desplazado hacia un solo lado + archivo final limpio -> fuerte indicador de desregistro/desfase tinta blanca-color en impresión;
- blanco uniforme alrededor de todos los píxeles -> revisar semitransparencias, residuos o halo del archivo.
Si el PNG ya tiene halo -> problema de preparación.
Si el PNG está limpio y el error aparece al imprimir -> impresión/desregistro gana peso.
No digas "100%" solo por una fotografía; usa "la causa más probable" o "es un indicador muy fuerte".

SEMITRANSPARENCIAS:
Un diseño individual puede estar limpio y adquirir semitransparencias nuevamente al escalar/componer/exportar en un editor externo. Por eso importa revisar la plantilla FINAL.

======================================================================
FORMA DE RESPONDER
======================================================================

- Español por defecto.
- Natural y directo.
- No sonar robótica.
- No dar una lista gigante si basta una ruta.
- Si el cliente es principiante, toma la decisión técnica por él cuando tengas datos suficientes.
- Si el usuario propone una técnica equivocada, corrígelo con explicación.
- Explica el porqué, no solo el nombre de una herramienta.
- No culpes a NOVAGE, al cliente o a la imprenta antes de descartar causas.
- No repitas una solución que el usuario ya dijo que no funcionó.

======================================================================
HERRAMIENTAS
======================================================================

Cuando una herramienta NOVAGE sea realmente útil, añade al final:
[HERRAMIENTA: Nombre exacto]

Nombres válidos:
Guía del Taller Digital
Mejorador con IA
Mejorador NOVAGE
Eliminador de Fondos
Semitonos
Semitransparencias
Contraer Bordes
Redimensionador
Crear Plantilla DTF
Armador de Plantilla DTF
Mockups
Pedrería
Calculadora de Precios
Escáner DTF
DTF + Pedrería
Conversor de Archivos
Guía de Medidas
DTF con Vinil
Marcos Grunge
Vectorizador NOVAGE
Semitonos Fáciles

Semitonos Profesionales puede mencionarse en texto, pero no emitas marcador hasta que exista una URL configurada.

======================================================================
ESTADO INTERNO
======================================================================

Al final, después de los marcadores de herramienta, añade exactamente uno:
[ESTADO: PREGUNTA] si todavía necesitas una respuesta decisiva antes de dar una solución fiable.
[ESTADO: SOLUCION] si ya diste una ruta, solución o diagnóstico accionable que el cliente puede probar/verificar.

No expliques estos marcadores. La interfaz los oculta.
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

    const {
      mensaje,
      imagen,
      previousResponseId,
      conversationContext,
    } = req.body || {};

    const mensajeLimpio = typeof mensaje === "string" ? mensaje.trim() : "";

    if (!mensajeLimpio && !imagen) {
      return res.status(400).json({ error: "Escribe un mensaje o sube una imagen." });
    }

    const recentConversation = cleanConversationContext(conversationContext);
    const retrievalQuery = buildRetrievalQuery(
      mensajeLimpio || "Analiza el archivo adjunto para preparación DTF.",
      recentConversation
    );

    const knowledgePromise = searchVectorStore(
      process.env.NOVAGE_VECTOR_STORE_ID,
      retrievalQuery,
      { maxResults: KNOWLEDGE_RESULTS }
    );

    const casesPromise = searchVectorStore(
      process.env.NOVAGE_CASES_VECTOR_STORE_ID,
      retrievalQuery,
      {
        maxResults: CASE_RESULTS,
        filters: { type: "eq", key: "resolved", value: true },
      }
    );

    const [knowledgeSettled, casesSettled] = await Promise.allSettled([
      knowledgePromise,
      casesPromise,
    ]);

    const knowledgeResults =
      knowledgeSettled.status === "fulfilled" ? knowledgeSettled.value : [];

    const caseResults =
      casesSettled.status === "fulfilled" ? casesSettled.value : [];

    if (knowledgeSettled.status === "rejected") {
      console.error("NOVAGE knowledge search error:", knowledgeSettled.reason);
    }

    if (casesSettled.status === "rejected") {
      console.error("NOVAGE cases search error:", casesSettled.reason);
    }

    const officialContext = formatSearchResults(knowledgeResults, {
      maxChars: 15000,
      minScore: 0,
    });

    const solvedCasesContext = formatSearchResults(caseResults, {
      maxChars: 9000,
      minScore: CASE_MIN_SCORE,
    });

    const instructions = `${NOVAGE_MASTER_INSTRUCTIONS}

======================================================================
CONTEXTO OFICIAL NOVAGE RECUPERADO PARA ESTA CONSULTA
======================================================================
${officialContext || "No se recuperaron fragmentos oficiales específicos para esta consulta. No inventes información NOVAGE que no esté respaldada."}

======================================================================
CASOS REALES CONFIRMADOS PARECIDOS
======================================================================
${solvedCasesContext || "No se recuperaron casos confirmados suficientemente parecidos. No fuerces analogías con casos anteriores."}

REGLA FINAL DE ESTA CONSULTA:
Usa el contexto oficial como fuente NOVAGE principal. Usa los casos solo para reconocer patrones. Luego aplica tu propio razonamiento profesional al caso actual.`;

    const content = [];

    if (mensajeLimpio) {
      content.push({ type: "input_text", text: mensajeLimpio });
    }

    if (imagen) {
      content.push({ type: "input_image", image_url: imagen });
    }

    const payload = {
      model: "gpt-6-luna",
      instructions,
      input: [{ role: "user", content }],
      reasoning: { effort: "medium" },
      max_output_tokens: 2400,
      store: true,
      prompt_cache_key: "novage-luna-v3",
      metadata: {
        app: "novage-ai",
        knowledge_version: "v3",
      },
    };

    if (previousResponseId) {
      payload.previous_response_id = previousResponseId;
    }

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
      respuesta: extractText(data),
      responseId: data.id || null,
      remaining: limit.remaining,
      retrieval: {
        knowledge: compactHitMetadata(knowledgeResults),
        cases: compactHitMetadata(caseResults, CASE_MIN_SCORE),
      },
    });
  } catch (error) {
    console.error("NOVAGE AI Error:", error);
    return res.status(500).json({
      error: "Ocurrió un error interno al procesar la solicitud.",
      details: error?.message || "Error desconocido",
    });
  }
}
