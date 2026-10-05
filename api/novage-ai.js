// NOVAGE AI • Luna • v4.2
const KNOWLEDGE_VERSION = "v4.2";
const RATE_LIMIT = 30;
const WINDOW_MS = 60 * 60 * 1000;
const KNOWLEDGE_RESULTS = 7;
const CASE_RESULTS = 5;
const CASE_MIN_SCORE = 0.28;
// Máximo de preguntas decisivas seguidas antes de obligar a resolver.
const MAX_QUESTIONS_IN_ROW = 3;
// Detecta color de prenda escrito en texto libre ("playera negra", "camisa de color blanco").
const GARMENT_COLOR_REGEX =
  /\b(playera|camiseta|camisa|prenda|sudadera|hoodie|polo|blusa|tela|gorra|bolsa)s?\s+(?:de\s+color\s+|color\s+)?(negr[ao]s?|blanc[ao]s?|gris(?:es)?|roj[ao]s?|azul(?:es)?|verdes?|amarill[ao]s?|rosas?|beige|crema|marino|vino|caf[eé])(?![a-záéíóúñ])/i;

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

  if (data?.status === "incomplete") {
    console.error("NOVAGE respuesta incompleta:", data?.incomplete_details);
  }

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
              key:
                typeof item.decision.key === "string"
                  ? item.decision.key.trim().slice(0, 80)
                  : "",
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

  // Genérico: cualquier respuesta por botón incluye "Variable: <key>" (canónicas y personalizadas).
  for (const match of normalized.matchAll(/^variable:\s*([a-z0-9_]+)\s*$/gm)) {
    if (match[1] !== "unknown") facts.add(match[1]);
  }

  return facts;
}

function parseDecisionAnswer(text) {
  const value = String(text || "");
  if (!/RESPUESTA A PREGUNTA DECISIVA/i.test(value)) return null;

  const key = value.match(/^Variable:\s*(.+)$/im)?.[1]?.trim() || "";
  const question = value.match(/^Pregunta:\s*(.+)$/im)?.[1]?.trim() || "";
  const answer = value.match(/^Respuesta seleccionada por el cliente:\s*(.+)$/im)?.[1]?.trim() || "";

  if (!question || !answer) return null;
  return { key, question, answer };
}

// Recorre la conversación + mensaje actual como una sola lista de turnos.
function allTurns(currentMessage, conversation) {
  return [...conversation, { role: "user", text: currentMessage || "" }];
}

function collectResolvedFacts(currentMessage, conversation) {
  const facts = new Set();
  const turns = allTurns(currentMessage, conversation);
  const all = turns.map((item) => item.text || "").join("\n");

  for (const fact of resolvedFactsFromText(all)) facts.add(fact);

  // Respuesta escrita a mano (sin botón) a una pregunta decisiva: también la resuelve.
  for (let i = 1; i < turns.length; i += 1) {
    const item = turns[i];
    const prev = turns[i - 1];
    if (
      item.role === "user" &&
      item.text &&
      !parseDecisionAnswer(item.text) &&
      prev?.role === "assistant" &&
      prev.state === "question" &&
      prev.decision?.key
    ) {
      facts.add(prev.decision.key);
    }
  }

  // El cliente ya escribió el color de la prenda.
  const userText = turns
    .filter((item) => item.role === "user")
    .map((item) => item.text || "")
    .join("\n");
  if (GARMENT_COLOR_REGEX.test(userText)) facts.add("garment_black");

  return facts;
}

// Hechos confirmados en lenguaje claro para que la solución NO los ignore.
function buildConfirmedFacts(currentMessage, conversation) {
  const facts = [];
  const turns = allTurns(currentMessage, conversation);

  for (let i = 0; i < turns.length; i += 1) {
    const item = turns[i];
    if (item.role !== "user" || !item.text) continue;

    const parsed = parseDecisionAnswer(item.text);
    if (parsed) {
      facts.push(`- ${parsed.question} → ${parsed.answer}`);
      continue;
    }

    const prev = turns[i - 1];
    if (prev?.role === "assistant" && prev.state === "question" && prev.decision?.question) {
      facts.push(`- ${prev.decision.question} → (respondió escribiendo) ${item.text.slice(0, 300)}`);
    }
  }

  return [...new Set(facts)];
}

function questionsInRow(conversation) {
  let count = 0;
  for (let i = conversation.length - 1; i >= 0; i -= 1) {
    const item = conversation[i];
    if (item.role !== "assistant") continue;
    if (item.state === "question") count += 1;
    else break;
  }
  return count;
}

// Limpia el texto técnico de respuestas por botón para no ensuciar la búsqueda.
function retrievalText(text) {
  const parsed = parseDecisionAnswer(text);
  if (parsed) return `${parsed.question} ${parsed.answer}`;
  return String(text || "");
}

function buildRetrievalQuery(message, conversation) {
  const firstUser = conversation.find((item) => item.role === "user");
  const recent = conversation
    .slice(-8)
    .map(
      (item) =>
        `${item.role === "assistant" ? "LUNA" : "CLIENTE"}: ${retrievalText(item.text || item.displayText).slice(0, 700)}`
    )
    .join("\n");

  return [
    "Consulta actual del cliente:",
    retrievalText(message),
    firstUser ? `\nProblema original:\n${retrievalText(firstUser.text).slice(0, 1500)}` : "",
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

function buildSharedContext(officialContext, solvedCasesContext, conversation, confirmedFacts = []) {
  return `
======================================================================
HECHOS CONFIRMADOS POR EL CLIENTE (tienen prioridad, no los vuelvas a preguntar)
======================================================================
${confirmedFacts.length ? confirmedFacts.join("\n") : "Ninguno todavía."}

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
- custom   (pregunta personalizada, ver regla 7)
- none

Y clasifica case_type:
- detailed_black_background: diseño detallado/texturizado donde el negro es parte de la composición.
- white_border_print: bordes/halo blanco tras imprimir DTF.
- pixelated_design: calidad, pixelado, mejorar vs vectorizar.
- dtf_application: planchado/aplicación DTF, se despega, se agrieta, polvo, tacto, migración.
- sublimation: sublimación (textil, tazas, rígidos): colores, ghosting, manchas, materiales.
- vinyl: vinil textil/HTV, corte, depilado, adherencia.
- file_preparation: fondos, semitransparencias, medidas, plantillas, formatos.
- business: precios, costos, márgenes, venta.
- other: saludo, pregunta general o fuera de tema.

PRINCIPIO:
Pregunta solo si la respuesta cambia realmente la ruta. Si Luna puede decidir técnicamente con lo que ve y sabe, devuelve none.
Un asesor experto NO pregunta por preguntar: si con lo que ya sabes puedes dar la causa más probable y un plan, devuelve none.

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
Preguntas teóricas ("¿qué es...?", "¿diferencia entre...?"), saludos y precios generales -> none.

7) PREGUNTA PERSONALIZADA (custom)
Usa missing_fact = custom SOLO cuando el caso NO encaja en las variables canónicas y existe UN dato que el cliente conoce, que tú no puedes observar ni deducir, y cuya respuesta cambia la causa o la ruta.
Ejemplos válidos:
- "Se me despegó el estampado" sin decir la técnica -> ¿Qué técnica usaste? | DTF | Sublimación | Vinil textil | Otra
- DTF que se despega sin saber cuándo -> ¿Cuándo se despegó? | Al retirar el film | En el primer lavado | Tras varios lavados
- Sublimación con colores apagados sin saber el material -> ¿De qué material es la prenda? | 100% poliéster | Mezcla con algodón | Algodón | No lo sé
Reglas de custom:
- custom_key: snake_case descriptivo y estable (ej. tecnica_usada, momento_despegue, material_prenda).
- custom_question: una sola pregunta corta y clara para principiante.
- custom_options: 2 a 4 opciones, máximo 5 palabras cada una.
- custom_message: 1 o 2 frases en tono asesor NOVAGE explicando POR QUÉ ese dato cambia el diagnóstico. Sin listar todas las ramas.
- Nunca preguntes qué herramienta o técnica QUIERE usar el cliente: eso lo decide Luna.
Si missing_fact NO es custom: custom_key, custom_question y custom_message = "" y custom_options = [].

8) DATOS DADOS EN TEXTO LIBRE
Si el cliente ya escribió el dato (ej. "es para playera negra", "la armé en Canva", "es 100% poliéster"), considéralo resuelto aunque no haya usado botones.

No escribas la pregunta canónica al usuario. El servidor la generará de forma canónica.
`;

const SOLUTION_INSTRUCTIONS = `
Eres Luna, asesora técnica senior de NOVAGE. Hablas como alguien que ha preparado e impreso miles de trabajos: DTF, sublimación, vinil textil, DTG, serigrafía básica, preparación de archivos, semitonos, semitransparencias, vectorización, fondos, plantillas y diagnóstico de errores de producción.

Tu trabajo NO es conversar: es DIAGNOSTICAR y decirle al cliente EXACTAMENTE qué hacer y cómo comprobarlo.

El DECISION GATE ya determinó que existe información suficiente. NO hagas preguntas. Si falta un dato menor, decláralo como supuesto en una línea ("Asumo que ...; si no es así, dímelo y ajusto.") y resuelve igual.

MÉTODO INTERNO (no lo escribas): OBSERVAR -> DESCARTAR -> DIAGNOSTICAR -> FLUJO -> COMPROBAR.

PRIORIDAD DE EVIDENCIA:
1. HECHOS CONFIRMADOS POR EL CLIENTE (sección del contexto). Nunca los contradigas ni los ignores.
2. Lo que observas en la imagen adjunta (puede venir de un turno anterior de la conversación).
3. Conocimiento oficial NOVAGE recuperado.
4. Casos reales confirmados parecidos (solo si coinciden las señales clave).
5. Conocimiento profesional general.

FORMATO OBLIGATORIO del campo message (títulos en **negritas**, sin #, sin tablas, sin emojis):

**Diagnóstico:** 1 a 3 frases. Causa más probable y POR QUÉ: qué señal concreta la delata. Si hay imagen, menciona algo específico que viste en ella.

**Qué hacer:**
1. Paso concreto que empieza con verbo (Sube, Descarga, Activa, Plancha, Revisa...). Nombra la herramienta NOVAGE o el ajuste exacto.
2. ...
(3 a 6 pasos, en el orden real de trabajo)

**Cómo comprobarlo:** 1 o 2 frases con la prueba rápida que confirma que quedó bien ANTES de producir en serie.

**Si no se corrige:** 1 frase con la siguiente causa más probable y qué revisar.

EXCEPCIÓN: preguntas teóricas o simples ("¿qué es...?", "¿diferencia entre...?") se responden directo en 2 a 4 frases, sin el formato completo.

TONO NOVAGE:
- Tutea. Directa, segura, cercana, cero relleno. Como un maestro de taller que sabe y lo explica fácil.
- Prohibido: "¡Claro!", "Excelente pregunta", "Espero que te sirva", "No dudes en...", "Como IA...", resumir lo que el cliente acaba de decir.
- No uses "depende" sin decir de qué depende y cuál es tu recomendación.
- Si el cliente propone una técnica inadecuada, corrígelo con el motivo técnico, sin regañar.

PARÁMETROS DE PRODUCCIÓN (temperatura, tiempo, presión, despegue):
- Da el RANGO DE REFERENCIA de la base de conocimiento como punto de partida.
- Aclara en la misma frase que se confirma con la ficha técnica del proveedor del film/papel/vinil y recomienda una prueba en retazo.
- Nunca te niegues a orientar y nunca presentes un rango como valor universal.

REGLAS:
- No le pidas al cliente que elija una técnica que tú puedes determinar.
- No inventes DPI, centímetros, resolución real, perfiles ni funciones de herramientas no documentadas.
- No culpes a NOVAGE, cliente o imprenta sin descartar causas.
- Si el cliente dijo que una solución anterior NO funcionó: no la repitas; explica qué causa descarta ese resultado y pasa a la siguiente.
- Usa herramientas NOVAGE solo cuando realmente ayudan; si el problema es de prensa o material, la solución puede no llevar herramienta.

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

CAMPO tools:
- Solo herramientas que aparecen en tus pasos, en el orden en que se usan, máximo 3.
- Usa "Crear Plantilla DTF" (no "Armador de Plantilla DTF").
- Si la solución no requiere herramienta NOVAGE, devuelve [].
- Semitonos Profesionales puede mencionarse en el texto aunque no exista tarjeta configurada.
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

function slugKey(value) {
  return String(value || "")
    .toLowerCase()
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 50);
}

// Valida la pregunta personalizada del Gate. Devuelve null si no sirve.
function buildCustomDecision(parsed, resolvedFacts) {
  const slug = slugKey(parsed?.custom_key);
  const question = String(parsed?.custom_question || "").trim().slice(0, 200);
  const message = String(parsed?.custom_message || "").trim().slice(0, 400);
  const options = [
    ...new Set(
      (Array.isArray(parsed?.custom_options) ? parsed.custom_options : [])
        .filter((value) => typeof value === "string")
        .map((value) => value.trim().slice(0, 60))
        .filter(Boolean)
    )
  ].slice(0, 4);

  if (!slug || !question || options.length < 2) return null;

  const key = `custom_${slug}`;
  if (resolvedFacts.has(key) || resolvedFacts.has(slug)) return null;

  return {
    key,
    message: message || "Necesito un solo dato para darte el diagnóstico correcto.",
    question,
    options
  };
}

async function runDecisionGate({ message, image, imageFromContext, sharedContext, resolvedFacts }) {
  const content = [
    {
      type: "input_text",
      text: [
        `Consulta actual del cliente:\n${message || "Analiza la imagen adjunta."}`,
        imageFromContext
          ? "\n[La imagen adjunta es el diseño que el cliente compartió antes en esta conversación.]"
          : "",
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
    reasoning: { effort: "low" },
    // El razonamiento consume tokens de salida: con 500 la respuesta se cortaba
    // y el Gate caía siempre en "none".
    max_output_tokens: 2500,
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
                "dtf_application",
                "sublimation",
                "vinyl",
                "file_preparation",
                "business",
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
                "custom",
                "none"
              ]
            },
            custom_key: { type: "string" },
            custom_message: { type: "string" },
            custom_question: { type: "string" },
            custom_options: {
              type: "array",
              items: { type: "string" }
            },
            confidence: {
              type: "string",
              enum: ["high", "medium", "low"]
            },
            reason: { type: "string" }
          },
          required: [
            "case_type",
            "missing_fact",
            "custom_key",
            "custom_message",
            "custom_question",
            "custom_options",
            "confidence",
            "reason"
          ],
          additionalProperties: false
        }
      }
    },
    store: false,
    prompt_cache_key: "novage-decision-gate-v4-2"
  });

  const parsed = parseJsonOutput(data, {
    case_type: "other",
    missing_fact: "none",
    custom_key: "",
    custom_message: "",
    custom_question: "",
    custom_options: [],
    confidence: "low",
    reason: "No se pudo clasificar."
  });

  if (resolvedFacts.has(parsed.missing_fact)) {
    parsed.missing_fact = "none";
  }

  return { parsed, responseId: data.id || null };
}

async function runSolution({ message, image, imageFromContext, sharedContext, gateInfo, forcedByLimit }) {
  const content = [
    {
      type: "input_text",
      text: [
        `Consulta actual del cliente:\n${message || "Analiza la imagen adjunta."}`,
        imageFromContext
          ? "\n[La imagen adjunta es el diseño que el cliente compartió antes en esta conversación. Analízala.]"
          : "",
        gateInfo
          ? `\nCLASIFICACIÓN PREVIA DEL CASO: ${gateInfo.caseType} | motivo: ${gateInfo.reason || "sin detalle"}`
          : "",
        forcedByLimit
          ? "\nYa se hicieron varias preguntas seguidas. Resuelve con lo disponible y declara tus supuestos."
          : "",
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
    // Incluye margen para el razonamiento; con 2200 la respuesta podía quedar truncada.
    max_output_tokens: 7000,
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
    prompt_cache_key: "novage-solution-v4-2",
    metadata: {
      app: "novage-ai",
      knowledge_version: KNOWLEDGE_VERSION
    }
  });

  const rawText = extractText(data);
  const parsed = parseJsonOutput(data, {
    // Nunca mostrar JSON roto al cliente.
    message: rawText && !rawText.startsWith("{")
      ? rawText
      : "No pude terminar la respuesta. Envíame tu mensaje de nuevo, por favor.",
    tools: []
  });

  const tools = Array.isArray(parsed.tools)
    ? [
        ...new Set(
          parsed.tools
            .filter((name) => TOOL_NAMES.includes(name))
            .map((name) => (name === "Armador de Plantilla DTF" ? "Crear Plantilla DTF" : name))
        )
      ].slice(0, 4)
    : [];

  return {
    message: typeof parsed.message === "string" && parsed.message.trim()
      ? parsed.message.trim()
      : "No se pudo generar una respuesta.",
    tools,
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

    const { mensaje, imagen: imagenRaw, imagenContexto, conversationContext } = req.body || {};
    const message = typeof mensaje === "string" ? mensaje.trim().slice(0, 6000) : "";
    const imagen =
      typeof imagenRaw === "string" && imagenRaw.startsWith("data:image/") ? imagenRaw : null;
    const imageFromContext = Boolean(imagen && imagenContexto === true);

    if (!message && !imagen) {
      return res.status(400).json({ error: "Escribe un mensaje o sube una imagen." });
    }

    const conversation = cleanConversationContext(conversationContext);
    const resolvedFacts = collectResolvedFacts(message, conversation);
    const confirmedFacts = buildConfirmedFacts(message, conversation);
    const forceSolution = questionsInRow(conversation) >= MAX_QUESTIONS_IN_ROW;

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
      conversation,
      confirmedFacts
    );

    // Si el Gate falla, no tumbamos la conversación: pasamos directo a resolver.
    let gate;
    try {
      gate = await runDecisionGate({
        message,
        image: imagen,
        imageFromContext,
        sharedContext,
        resolvedFacts
      });
    } catch (gateError) {
      console.error("NOVAGE gate error:", gateError);
      gate = {
        parsed: { case_type: "other", missing_fact: "none", confidence: "low", reason: "" },
        responseId: null
      };
    }

    let missingFact = forceSolution ? "none" : gate.parsed?.missing_fact || "none";

    // Guardrail determinista: diseño detallado con fondo negro exige conocer el
    // color de la prenda, salvo que ya esté resuelto (botón o texto libre).
    if (
      !forceSolution &&
      gate.parsed?.case_type === "detailed_black_background" &&
      !resolvedFacts.has("garment_black")
    ) {
      missingFact = "garment_black";
    }

    let decision = null;
    if (missingFact === "custom") {
      decision = buildCustomDecision(gate.parsed, resolvedFacts);
    } else if (missingFact !== "none" && DECISIONS[missingFact]) {
      decision = { key: missingFact, ...DECISIONS[missingFact] };
    }

    if (decision) {
      return res.status(200).json({
        respuesta: decision.message,
        estado: "question",
        decision: {
          key: decision.key,
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
      imageFromContext,
      sharedContext,
      gateInfo: {
        caseType: gate.parsed?.case_type || "other",
        reason: gate.parsed?.reason || ""
      },
      forcedByLimit: forceSolution
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
