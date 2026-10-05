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

function parseStructuredAnswer(data) {
  const raw = extractText(data);

  try {
    const parsed = JSON.parse(raw);
    const status = parsed?.status === "question" ? "question" : "solution";
    const message = typeof parsed?.message === "string"
      ? parsed.message.trim()
      : "No se pudo obtener una respuesta.";

    const question = typeof parsed?.decision?.question === "string"
      ? parsed.decision.question.trim()
      : "";

    const options = Array.isArray(parsed?.decision?.options)
      ? parsed.decision.options
          .filter((item) => typeof item === "string" && item.trim())
          .map((item) => item.trim())
          .slice(0, 4)
      : [];

    const tools = Array.isArray(parsed?.tools)
      ? [...new Set(
          parsed.tools
            .filter((item) => typeof item === "string" && item.trim())
            .map((item) => item.trim())
        )]
      : [];

    if (status === "question" && (!question || options.length < 2)) {
      return {
        message,
        status: "solution",
        decision: { question: "", options: [] },
        tools,
      };
    }

    return {
      message,
      status,
      decision:
        status === "question"
          ? { question, options }
          : { question: "", options: [] },
      tools,
    };
  } catch (error) {
    console.error("NOVAGE structured output parse error:", error, raw);
    return {
      message: raw,
      status: "solution",
      decision: { question: "", options: [] },
      tools: [],
    };
  }
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
            .replace(/\[PREGUNTA_DECISIVA:\s*[^\]]+\]/gi, "")
            .replace(/\[OPCIONES:\s*[^\]]+\]/gi, "")
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

Haz como máximo 1 pregunta decisiva por turno.
No preguntes algo que ya puedas observar en la imagen.
En cuanto tengas evidencia suficiente, deja de preguntar y resuelve.


======================================================================
PROTOCOLO DE PREGUNTAS DECISIVAS CON BOTONES
======================================================================

La respuesta final se entrega al frontend en una estructura JSON. NO escribas marcadores técnicos dentro del texto visible.

Cuando falte UN dato que cambie realmente el diagnóstico o la ruta:
- status = "question"
- message = explicación breve de por qué ese dato importa, SIN repetir la pregunta;
- decision.question = una sola pregunta decisiva;
- decision.options = entre 2 y 4 opciones cortas;
- tools = [] salvo que una herramienta sea útil solo como referencia y no implique una solución prematura.

Cuando ya exista información suficiente:
- status = "solution"
- message = diagnóstico o flujo concreto, ordenado y accionable;
- decision.question = "";
- decision.options = [];
- tools = únicamente las herramientas NOVAGE realmente útiles.

REGLAS:
- Máximo UNA pregunta decisiva por respuesta.
- No preguntes al usuario qué técnica desea utilizar si tú puedes decidirlo.
- No preguntes cosas evidentes en la imagen.
- No repitas una pregunta ya contestada.
- Si una opción seleccionada ya resuelve la incertidumbre, deja de preguntar y da el flujo exacto.
- No muestres todas las ramas posibles antes de que el usuario conteste. Eso vuelve la respuesta confusa.
- Las opciones deben estar escritas para principiantes, no para técnicos.

EJEMPLOS DE DECISIONES:
Pregunta: ¿La playera será negra?
Opciones: Sí | No

Pregunta: ¿Dónde creaste la plantilla final?
Opciones: Crear Plantilla NOVAGE | Canva | Photoshop | Otra aplicación

Pregunta: ¿Cómo aparece el borde blanco?
Opciones: Principalmente de un solo lado | Rodea todo el contorno | No estoy seguro

CASO MODELO - DISEÑO DETALLADO DE PINTEREST CON FONDO NEGRO:
Primero reconoce que:
- es rasterizado, detallado y texturizado;
- vectorizar no es la primera opción;
- el negro funciona como parte visual o espacio negativo;
- la decisión que cambia la preparación es el color de la prenda.

La primera respuesta ideal tiene:
status = "question"
message = "Es un diseño detallado y texturizado, así que no lo vectorizaría como primer paso. El dato que cambia la preparación es el color de la prenda."
decision.question = "¿La playera será negra?"
decision.options = ["Sí", "No"]

Si responde "Sí":
- Aprovecha el negro de la tela.
- Si la calidad lo necesita, primero Mejorador con IA.
- El cliente debe descargar el archivo mejorado.
- Después debe pasarlo a Semitonos Fáciles o Semitonos Profesionales.
- Evita imprimir una gran plasta negra.
- Da los pasos en orden.
- status = "solution".

Si responde "No":
- Explica que el diseño depende visualmente del negro.
- No prometas que quedará igual.
- Recomienda adaptar/rediseñar el arte antes de imprimirlo.
- No elimines el negro automáticamente si destruye la composición.
- status = "solution" si ya diste una ruta accionable.


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

La salida estructurada tiene un array "tools".
Incluye únicamente nombres exactos de herramientas configuradas cuando sean realmente útiles.

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

Semitonos Profesionales puede mencionarse dentro de "message", pero no lo añadas a "tools" hasta que exista una URL configurada.

No metas herramientas por rellenar.
Si estás esperando una respuesta decisiva y la herramienta depende de esa respuesta, deja "tools" vacío.

======================================================================
ESTADO INTERNO
======================================================================

La estructura de salida debe representar exactamente uno de dos estados:

QUESTION:
- status = "question"
- decision.question no vacío
- decision.options con al menos 2 opciones
- todavía falta un dato decisivo

SOLUTION:
- status = "solution"
- decision.question = ""
- decision.options = []
- ya diste una ruta, diagnóstico o solución que el cliente puede probar

El campo "message" es lo único que verá el cliente como texto de Luna.
No pongas JSON, marcadores internos ni nombres de campos dentro de "message".
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
      content.push({ type: "input_image", image_url: imagen, detail: "high" });
    }

    const payload = {
      model: "gpt-6-luna",
      instructions,
      input: [{ role: "user", content }],
      reasoning: { effort: "medium" },
      max_output_tokens: 2400,
      text: {
        format: {
          type: "json_schema",
          name: "novage_luna_response",
          strict: true,
          schema: {
            type: "object",
            properties: {
              message: { type: "string" },
              status: {
                type: "string",
                enum: ["question", "solution"]
              },
              decision: {
                type: "object",
                properties: {
                  question: { type: "string" },
                  options: {
                    type: "array",
                    items: { type: "string" }
                  }
                },
                required: ["question", "options"],
                additionalProperties: false
              },
              tools: {
                type: "array",
                items: {
                  type: "string",
                  enum: [
                    "Guía del Taller Digital",
                    "Mejorador con IA",
                    "Mejorador NOVAGE",
                    "Eliminador de Fondos",
                    "Semitonos",
                    "Semitransparencias",
                    "Contraer Bordes",
                    "Redimensionador",
                    "Crear Plantilla DTF",
                    "Armador de Plantilla DTF",
                    "Mockups",
                    "Pedrería",
                    "Calculadora de Precios",
                    "Escáner DTF",
                    "DTF + Pedrería",
                    "Conversor de Archivos",
                    "Guía de Medidas",
                    "DTF con Vinil",
                    "Marcos Grunge",
                    "Vectorizador NOVAGE",
                    "Semitonos Fáciles"
                  ]
                }
              }
            },
            required: ["message", "status", "decision", "tools"],
            additionalProperties: false
          }
        }
      },
      store: true,
      prompt_cache_key: "novage-luna-v4",
      metadata: {
        app: "novage-ai",
        knowledge_version: "v4",
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

    const structured = parseStructuredAnswer(data);

    return res.status(200).json({
      respuesta: structured.message,
      estado: structured.status,
      decision: structured.decision,
      tools: structured.tools,
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
