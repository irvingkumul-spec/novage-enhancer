const RATE_LIMIT = 30;
const WINDOW_MS = 60 * 60 * 1000;
const KNOWLEDGE_RESULTS = 7;
const CASE_RESULTS = 5;
const CASE_MIN_SCORE = 0.28;

const usage = globalThis.__NOVAGE_AI_USAGE__ || new Map();
globalThis.__NOVAGE_AI_USAGE__ = usage;

const TOOL_NAMES = [
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
];

const DECISIONS = {
  garment_black: {
    message:
      "Es un diseño detallado y texturizado, así que no lo vectorizaría como primer paso. Para elegir la preparación correcta solo necesito confirmar el color de la prenda.",
    question: "¿La playera será negra?",
    options: ["Sí", "No"]
  },
  template_source: {
    message:
      "Antes de atribuir el borde blanco a los semitonos o a la impresión, necesito reconstruir cómo se preparó la plantilla final.",
    question: "¿Dónde creaste la plantilla final de impresión?",
    options: ["Crear Plantilla NOVAGE", "Canva", "Photoshop", "Otra aplicación"]
  },
  final_scan: {
    message:
      "Como la plantilla se creó fuera de NOVAGE, lo importante es saber si revisaste el archivo final ya exportado, porque al componer o exportar pueden aparecer semitransparencias nuevamente.",
    question: "¿Pasaste la plantilla FINAL por Semitransparencias o Escáner DTF?",
    options: ["Sí", "No", "No estoy seguro"]
  },
  white_border_pattern: {
    message:
      "Ya tenemos suficiente información del flujo. Ahora necesito distinguir si el blanco se comporta como un residuo del archivo o como un desregistro de la tinta blanca.",
    question: "¿Cómo aparece el borde blanco en la impresión?",
    options: ["Principalmente de un solo lado", "Rodea todo el contorno", "No estoy seguro"]
  },
  design_type: {
    message:
      "Para decidir correctamente entre mejorar y vectorizar necesito saber qué tipo de diseño estás trabajando.",
    question: "¿Qué tipo de diseño es?",
    options: ["Logo o texto", "Foto / anime / ilustración detallada", "Gráfico plano", "No estoy seguro"]
  }
};

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

  return texts.join("\n").trim();
}

function parseJsonOutput(data, fallback = {}) {
  const raw = extractText(data);

  try {
    return JSON.parse(raw);
  } catch (error) {
    console.error("NOVAGE JSON parse error:", error, raw);
    return fallback;
  }
}

function cleanConversationContext(value) {
  if (!Array.isArray(value)) return [];

  return value
    .slice(-14)
    .map((item) => ({
      role: item?.role === "assistant" ? "assistant" : "user",
      text:
        typeof item?.text === "string"
          ? item.text.trim().slice(0, 2600)
          : "",
      displayText:
        typeof item?.displayText === "string"
          ? item.displayText.trim().slice(0, 1000)
          : "",
      hasImage: Boolean(item?.hasImage),
      state:
        item?.state === "question"
          ? "question"
          : item?.state === "solution"
          ? "solution"
          : null,
      decision:
        item?.decision &&
        typeof item.decision.question === "string" &&
        Array.isArray(item.decision.options)
          ? {
              question: item.decision.question.trim().slice(0, 500),
              options: item.decision.options
                .filter((x) => typeof x === "string")
                .map((x) => x.trim().slice(0, 180))
                .slice(0, 4)
            }
          : null
    }))
    .filter((item) => item.text || item.displayText);
}

function conversationAsText(conversation) {
  return conversation
    .map((item) => {
      const label = item.role === "assistant" ? "LUNA" : "CLIENTE";
      const image = item.hasImage ? " [adjuntó imagen]" : "";
      const state = item.state ? ` [estado=${item.state}]` : "";
      const decision = item.decision?.question
        ? `\nPregunta decisiva previa: ${item.decision.question} | opciones: ${item.decision.options.join(" | ")}`
        : "";
      return `${label}${image}${state}: ${item.text || item.displayText}${decision}`;
    })
    .join("\n");
}

function resolvedFactsFromText(text) {
  const normalized = String(text || "").toLowerCase();
  const facts = new Set();

  if (normalized.includes("pregunta: ¿la playera será negra?")) facts.add("garment_black");
  if (normalized.includes("pregunta: ¿dónde creaste la plantilla final de impresión?")) facts.add("template_source");
  if (normalized.includes("pregunta: ¿pasaste la plantilla final por semitransparencias o escáner dtf?")) facts.add("final_scan");
  if (normalized.includes("pregunta: ¿cómo aparece el borde blanco en la impresión?")) facts.add("white_border_pattern");
  if (normalized.includes("pregunta: ¿qué tipo de diseño es?")) facts.add("design_type");

  return facts;
}

function collectResolvedFacts(currentMessage, conversation) {
  const facts = new Set();
  const all = [currentMessage, ...conversation.map((item) => item.text || "")].join("\n");

  for (const fact of resolvedFactsFromText(all)) facts.add(fact);
  return facts;
}

function buildRetrievalQuery(message, conversation) {
  const recent = conversationAsText(conversation.slice(-8));

  return [
    "Consulta actual del cliente:",
    message,
    recent ? `\nContexto reciente:\n${recent}` : "",
    "\nBusca conocimiento útil para diagnosticar, elegir la técnica NOVAGE correcta y descartar causas."
  ]
    .filter(Boolean)
    .join("\n")
    .slice(0, 14000);
}

async function searchVectorStore(vectorStoreId, query, options = {}) {
  const body = {
    query,
    max_num_results: options.maxResults || 5,
    rewrite_query: true
  };

  if (options.filters) body.filters = options.filters;

  const response = await fetch(
    `https://api.openai.com/v1/vector_stores/${encodeURIComponent(vectorStoreId)}/search`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${process.env.OPENAI_API_KEY}`
      },
      body: JSON.stringify(body)
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
      attributes: item?.attributes || null
    }));
}

function buildSharedContext(officialContext, solvedCasesContext, conversation) {
  return `
======================================================================
CONTEXTO OFICIAL NOVAGE RECUPERADO
======================================================================
${officialContext || "No se recuperaron fragmentos oficiales específicos. No inventes información NOVAGE."}

======================================================================
CASOS REALES CONFIRMADOS PARECIDOS
======================================================================
${solvedCasesContext || "No se recuperaron casos confirmados suficientemente parecidos. No fuerces analogías."}

======================================================================
CONTEXTO RECIENTE DE LA CONVERSACIÓN
======================================================================
${conversationAsText(conversation) || "Sin contexto previo."}
`;
}

const GATE_INSTRUCTIONS = `
Eres el DECISION GATE de Luna, NOVAGE AI.
NO das la solución. Tu única tarea es decidir si falta UN dato que cambiaría materialmente el diagnóstico o el flujo recomendado.

Devuelve un campo missing_fact usando SOLO uno de estos valores:
- garment_black
- template_source
- final_scan
- white_border_pattern
- design_type
- none

PRINCIPIO:
Pregunta solo si la respuesta cambia realmente la ruta. Si Luna puede decidir técnicamente con lo que ve y sabe, devuelve none.

REGLAS OBLIGATORIAS:

1) DISEÑO DETALLADO / TEXTURIZADO / PINTEREST / FONDO NEGRO
Si observas un diseño rasterizado, detallado o texturizado donde el negro forma parte importante de la composición o funciona como espacio negativo, y todavía NO está confirmado si la playera será negra:
missing_fact = garment_black
No preguntes si quiere quitar el fondo. No preguntes qué técnica quiere usar.

Si ya está confirmado que la playera será negra o no, NO vuelvas a preguntar garment_black.

2) BORDES BLANCOS DESPUÉS DE IMPRIMIR
Si el usuario reporta bordes blancos y todavía no sabes dónde creó la plantilla final:
missing_fact = template_source

Si ya sabes que la plantilla se creó en Canva, Photoshop u otra aplicación externa y todavía no sabes si revisó LA PLANTILLA FINAL exportada con Semitransparencias o Escáner DTF:
missing_fact = final_scan

Si la plantilla fue creada con Crear Plantilla NOVAGE, o si fue externa pero ya confirmó que la plantilla FINAL fue revisada/limpiada, y todavía no sabes si el blanco aparece de un solo lado o rodea todo:
missing_fact = white_border_pattern

3) DISEÑO PIXELADO SIN IMAGEN NI TIPO CLARO
Si el usuario pregunta si debe mejorar o vectorizar, pero no hay imagen y no sabes si es logo/texto/plano o foto/anime/detallado:
missing_fact = design_type

Si hay imagen suficiente para identificar el tipo, NO preguntes design_type.

4) NO REPETIR
Si el contexto contiene una respuesta explícita a una pregunta decisiva previa, esa variable ya está resuelta. No la vuelvas a preguntar.

5) MÁXIMO UNA PREGUNTA
Nunca intentes pedir dos datos en el mismo turno.

6) SI NO FALTA UN DATO QUE CAMBIE LA RUTA
missing_fact = none

No escribas la pregunta al usuario. El servidor la generará de forma canónica.
`;

const SOLUTION_INSTRUCTIONS = `
Eres Luna, NOVAGE AI, especialista senior de NOVAGE en diseño gráfico aplicado a impresión, DTF, textiles, semitonos, semitransparencias, resolución, vectorización, fondos, contornos, plantillas y diagnóstico de errores.

El DECISION GATE ya determinó que existe información suficiente. NO hagas otra pregunta. Debes resolver.

MÉTODO:
PROBLEMA -> OBSERVAR -> DESCARTAR -> DIAGNÓSTICO -> FLUJO NOVAGE -> SIGUIENTE PASO.

PRIORIDAD:
1. Lo que observas en el archivo/imagen actual.
2. Lo que el cliente ya confirmó.
3. Conocimiento oficial NOVAGE recuperado.
4. Casos reales confirmados parecidos.
5. Conocimiento profesional general.

REGLAS:
- No le pidas al cliente que elija una técnica que tú puedes determinar.
- Corrige al usuario si propone una técnica poco adecuada.
- No inventes DPI, centímetros, resolución real, perfiles ni funciones no documentadas.
- No culpes a NOVAGE, cliente o imprenta sin descartar causas.
- No repitas una solución que ya se indicó como fallida.
- Da pasos en orden, concretos y accionables.
- Usa herramientas NOVAGE solo cuando realmente ayudan.

CASO CLAVE: DISEÑO DETALLADO DE INTERNET/PINTEREST CON FONDO NEGRO + PLAYERA NEGRA CONFIRMADA
- No vectorizar como primera opción.
- Aprovechar el negro de la tela para evitar imprimir una gran plasta negra.
- Si el usuario indicó que la imagen viene de Pinterest/internet, el PRIMER paso debe ser Mejorador con IA para trabajar desde una versión mejorada. Indica que descargue ese archivo mejorado.
- Después pasar el archivo mejorado por Semitonos Fáciles para un flujo sencillo, o Semitonos Profesionales si necesita mayor control.
- Explica que el objetivo es conservar textura/detalle y aprovechar el negro de la prenda.
- Da los pasos en orden 1, 2, 3, 4.

CASO CLAVE: DISEÑO DETALLADO CON FONDO NEGRO + PLAYERA NO NEGRA CONFIRMADA
- Explica que la composición depende visualmente del negro.
- No prometas que quitar el fondo dejará el mismo resultado.
- Recomienda adaptar/rediseñar el arte para esa prenda antes de imprimir.

CASO CLAVE: BORDES BLANCOS
- Si plantilla NOVAGE + blanco principalmente de un lado: fuerte indicador de desfase/desregistro tinta blanca-color.
- Si plantilla externa no revisada al final: semitransparencias del archivo final ganan probabilidad.
- Si plantilla externa revisada/limpia + blanco de un lado: desfase de tinta blanca gana probabilidad.
- Si blanco rodea uniformemente todos los píxeles: revisar semitransparencias, residuos o halo del archivo.
- No digas 100% solo por foto; usa causa más probable / indicador fuerte.

DIFERENCIAS CLAVE:
- foto/anime/arte complejo/texturas -> Mejorador con IA.
- logo/texto/gráfico plano/pocos colores -> Vectorizador NOVAGE.
- fondo ajeno separable -> Eliminador de Fondos.
- humo/sombra/degradado integrado -> Semitonos.
- Guía de Medidas decide; Redimensionador aplica.
- Semitonos Fáciles prioriza rapidez; Semitonos Profesionales mayor control.

FORMATO:
- Español natural, NOVAGE, directo.
- 2 a 5 párrafos o una secuencia corta numerada.
- No hagas preguntas en esta etapa.
- El campo tools debe contener solo herramientas con tarjeta configurada.
- Puedes mencionar Semitonos Profesionales en el texto aunque no exista tarjeta configurada.
`;

async function callOpenAI(payload) {
  const response = await fetch("https://api.openai.com/v1/responses", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${process.env.OPENAI_API_KEY}`
    },
    body: JSON.stringify(payload)
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data?.error?.message || "OpenAI no pudo generar una respuesta.");
  }

  return data;
}

async function runDecisionGate({ message, image, sharedContext, resolvedFacts }) {
  const content = [
    {
      type: "input_text",
      text: [
        `Consulta actual del cliente:\n${message || "Analiza la imagen adjunta."}`,
        `\nVariables decisivas ya resueltas: ${[...resolvedFacts].join(", ") || "ninguna"}`,
        sharedContext
      ].join("\n")
    }
  ];

  if (image) {
    content.push({ type: "input_image", image_url: image, detail: "high" });
  }

  const data = await callOpenAI({
    model: "gpt-6-luna",
    instructions: GATE_INSTRUCTIONS,
    input: [{ role: "user", content }],
    reasoning: { effort: "medium" },
    max_output_tokens: 500,
    text: {
      format: {
        type: "json_schema",
        name: "novage_decision_gate",
        strict: true,
        schema: {
          type: "object",
          properties: {
            case_type: {
              type: "string",
              enum: [
                "detailed_black_background",
                "white_border_print",
                "pixelated_design",
                "other"
              ]
            },
            missing_fact: {
              type: "string",
              enum: [
                "garment_black",
                "template_source",
                "final_scan",
                "white_border_pattern",
                "design_type",
                "none"
              ]
            },
            confidence: {
              type: "string",
              enum: ["high", "medium", "low"]
            },
            reason: { type: "string" }
          },
          required: ["case_type", "missing_fact", "confidence", "reason"],
          additionalProperties: false
        }
      }
    },
    store: false,
    prompt_cache_key: "novage-decision-gate-v4-1"
  });

  const parsed = parseJsonOutput(data, {
    case_type: "other",
    missing_fact: "none",
    confidence: "low",
    reason: "No se pudo clasificar."
  });

  if (resolvedFacts.has(parsed.missing_fact)) {
    parsed.missing_fact = "none";
  }

  return { parsed, responseId: data.id || null };
}

async function runSolution({ message, image, sharedContext }) {
  const content = [
    {
      type: "input_text",
      text: [
        `Consulta actual del cliente:\n${message || "Analiza la imagen adjunta."}`,
        sharedContext
      ].join("\n")
    }
  ];

  if (image) {
    content.push({ type: "input_image", image_url: image, detail: "high" });
  }

  const data = await callOpenAI({
    model: "gpt-6-luna",
    instructions: SOLUTION_INSTRUCTIONS,
    input: [{ role: "user", content }],
    reasoning: { effort: "medium" },
    max_output_tokens: 2200,
    text: {
      format: {
        type: "json_schema",
        name: "novage_solution",
        strict: true,
        schema: {
          type: "object",
          properties: {
            message: { type: "string" },
            tools: {
              type: "array",
              items: {
                type: "string",
                enum: TOOL_NAMES
              }
            }
          },
          required: ["message", "tools"],
          additionalProperties: false
        }
      }
    },
    store: true,
    prompt_cache_key: "novage-solution-v4-1",
    metadata: {
      app: "novage-ai",
      knowledge_version: "v4.1"
    }
  });

  const parsed = parseJsonOutput(data, {
    message: extractText(data) || "No se pudo generar una respuesta.",
    tools: []
  });

  return {
    message: typeof parsed.message === "string" ? parsed.message.trim() : "No se pudo generar una respuesta.",
    tools: Array.isArray(parsed.tools)
      ? [...new Set(parsed.tools.filter((name) => TOOL_NAMES.includes(name)))]
      : [],
    responseId: data.id || null
  };
}

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
        error: "Has alcanzado el límite temporal de NOVAGE AI. Intenta nuevamente más tarde."
      });
    }

    const { mensaje, imagen, conversationContext } = req.body || {};
    const message = typeof mensaje === "string" ? mensaje.trim() : "";

    if (!message && !imagen) {
      return res.status(400).json({ error: "Escribe un mensaje o sube una imagen." });
    }

    const conversation = cleanConversationContext(conversationContext);
    const resolvedFacts = collectResolvedFacts(message, conversation);

    const retrievalQuery = buildRetrievalQuery(
      message || "Analiza el archivo adjunto para preparación DTF.",
      conversation
    );

    const [knowledgeSettled, casesSettled] = await Promise.allSettled([
      searchVectorStore(process.env.NOVAGE_VECTOR_STORE_ID, retrievalQuery, {
        maxResults: KNOWLEDGE_RESULTS
      }),
      searchVectorStore(process.env.NOVAGE_CASES_VECTOR_STORE_ID, retrievalQuery, {
        maxResults: CASE_RESULTS,
        filters: { type: "eq", key: "resolved", value: true }
      })
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
      minScore: 0
    });

    const solvedCasesContext = formatSearchResults(caseResults, {
      maxChars: 9000,
      minScore: CASE_MIN_SCORE
    });

    const sharedContext = buildSharedContext(
      officialContext,
      solvedCasesContext,
      conversation
    );

    const gate = await runDecisionGate({
      message,
      image: imagen,
      sharedContext,
      resolvedFacts
    });

    let missingFact = gate.parsed?.missing_fact || "none";

    // Guardrail determinista: si el Gate reconoce el caso de diseño detallado
    // con fondo negro, el color de la prenda es obligatorio antes de resolver.
    if (
      gate.parsed?.case_type === "detailed_black_background" &&
      !resolvedFacts.has("garment_black")
    ) {
      missingFact = "garment_black";
    }

    if (missingFact !== "none" && DECISIONS[missingFact]) {
      const decision = DECISIONS[missingFact];

      return res.status(200).json({
        respuesta: decision.message,
        estado: "question",
        decision: {
          key: missingFact,
          question: decision.question,
          options: decision.options
        },
        tools: [],
        responseId: gate.responseId || `gate-${Date.now()}`,
        remaining: limit.remaining,
        gate: {
          caseType: gate.parsed?.case_type || "other",
          confidence: gate.parsed?.confidence || "low"
        },
        retrieval: {
          knowledge: compactHitMetadata(knowledgeResults),
          cases: compactHitMetadata(caseResults, CASE_MIN_SCORE)
        }
      });
    }

    const solution = await runSolution({
      message,
      image: imagen,
      sharedContext
    });

    return res.status(200).json({
      respuesta: solution.message,
      estado: "solution",
      decision: { key: "", question: "", options: [] },
      tools: solution.tools,
      responseId: solution.responseId,
      remaining: limit.remaining,
      gate: {
        caseType: gate.parsed?.case_type || "other",
        confidence: gate.parsed?.confidence || "low"
      },
      retrieval: {
        knowledge: compactHitMetadata(knowledgeResults),
        cases: compactHitMetadata(caseResults, CASE_MIN_SCORE)
      }
    });
  } catch (error) {
    console.error("NOVAGE AI Error:", error);

    return res.status(500).json({
      error: "Ocurrió un error interno al procesar la solicitud.",
      details: error?.message || "Error desconocido"
    });
  }
}
