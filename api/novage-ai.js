import crypto from "crypto";

// NOVAGE AI • Shopify / Vercel backend protegido • v5.1
// Sin guardado de chats. Requiere sesión firmada + origen oficial.
// Mantiene solo el contexto de la conversación ACTUAL que envía el frontend.

const RATE_LIMIT = 30;
const WINDOW_MS = 60 * 60 * 1000;
const KNOWLEDGE_RESULTS = 7;

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
      "El color de la prenda cambia por completo cómo conviene preparar este diseño.",
    question: "¿La playera será negra?",
    options: ["Sí", "No"]
  },
  template_source: {
    message:
      "Antes de culpar al archivo o a la impresión, necesito saber dónde se armó la plantilla final.",
    question: "¿Dónde creaste la plantilla final?",
    options: ["NOVAGE", "Canva", "Photoshop", "Otra app"]
  },
  final_scan: {
    message:
      "Si la plantilla se creó fuera de NOVAGE, importa saber si revisaste el archivo final exportado.",
    question: "¿Revisaste la plantilla FINAL con Semitransparencias o Escáner DTF?",
    options: ["Sí", "No", "No estoy seguro"]
  },
  white_border_pattern: {
    message:
      "La forma en que aparece el blanco ayuda a saber si viene del archivo o de la impresión.",
    question: "¿Cómo aparece el borde blanco?",
    options: ["De un solo lado", "Alrededor de todo", "No estoy seguro"]
  },
  design_type: {
    message:
      "El tipo de diseño define si conviene mejorar o vectorizar.",
    question: "¿Qué tipo de diseño es?",
    options: ["Logo o texto", "Foto / anime / ilustración", "Gráfico plano", "No estoy seguro"]
  }
};

const GARMENT_COLOR_REGEX =
  /\b(playera|camiseta|camisa|prenda|sudadera|hoodie|polo|blusa|tela|gorra|bolsa)s?\s+(?:de\s+color\s+|color\s+)?(negr[ao]s?|blanc[ao]s?|gris(?:es)?|roj[ao]s?|azul(?:es)?|verdes?|amarill[ao]s?|rosas?|beige|crema|marino|vino|caf[eé])(?![a-záéíóúñ])/i;

const DISSATISFACTION_REGEX =
  /\b(no\s+me\s+(?:resolvi[oó]|sirvi[oó]|ayud[oó])|no\s+lo\s+(?:resolvi[oó]|solucion[oó])|sigue\s+igual|no\s+funcion[oó])\b/i;

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

const ALLOWED_ORIGINS = new Set([
  "https://novage.store",
  "https://www.novage.store"
]);

const API_HOST = "tools.novage.store";

function getOrigin(req) {
  return String(req.headers.origin || "").trim();
}

function getHost(req) {
  return String(req.headers["x-forwarded-host"] || req.headers.host || "")
    .split(",")[0]
    .trim()
    .toLowerCase();
}

function isAllowedOrigin(origin) {
  return ALLOWED_ORIGINS.has(String(origin || "").trim());
}

function getUserAgent(req) {
  return String(req.headers["user-agent"] || "").slice(0, 600);
}

function fingerprint(req) {
  const ip = getClientIp(req);
  const ua = getUserAgent(req);

  return crypto
    .createHash("sha256")
    .update(`${ip}|${ua}`)
    .digest("base64url");
}

function safeEqual(a, b) {
  const aBuf = Buffer.from(String(a || ""));
  const bBuf = Buffer.from(String(b || ""));

  if (aBuf.length !== bBuf.length) return false;

  return crypto.timingSafeEqual(aBuf, bBuf);
}

function verifySessionToken(req) {
  const secret = process.env.NOVAGE_SESSION_SECRET;

  if (!secret) {
    throw new Error("NOVAGE_SESSION_SECRET no está configurada.");
  }

  const authorization = String(req.headers.authorization || "");
  const match = authorization.match(/^Bearer\s+(.+)$/i);

  if (!match) {
    throw new Error("Falta la sesión protegida.");
  }

  const token = match[1].trim();
  const parts = token.split(".");

  if (parts.length !== 2) {
    throw new Error("Sesión inválida.");
  }

  const [body, signature] = parts;

  const expected = crypto
    .createHmac("sha256", secret)
    .update(body)
    .digest("base64url");

  if (!safeEqual(signature, expected)) {
    throw new Error("Firma de sesión inválida.");
  }

  let payload;

  try {
    payload = JSON.parse(
      Buffer.from(body, "base64url").toString("utf8")
    );
  } catch {
    throw new Error("Sesión dañada.");
  }

  const now = Math.floor(Date.now() / 1000);

  if (payload?.aud !== "novage-ai") {
    throw new Error("Sesión no válida para NOVAGE AI.");
  }

  if (!Number.isFinite(payload?.exp) || payload.exp <= now) {
    throw new Error("La sesión expiró.");
  }

  if (!payload?.fp || payload.fp !== fingerprint(req)) {
    throw new Error("La sesión no corresponde a este dispositivo.");
  }

  return payload;
}

function setCors(req, res) {
  const origin = getOrigin(req);

  if (isAllowedOrigin(origin)) {
    res.setHeader("Access-Control-Allow-Origin", origin);
    res.setHeader("Vary", "Origin");
  }

  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader(
    "Access-Control-Allow-Headers",
    "Content-Type, Authorization"
  );
  res.setHeader("Access-Control-Max-Age", "600");
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
      const label = item.role === "assistant" ? "NOVAGE" : "CLIENTE";
      const image = item.hasImage ? " [adjuntó imagen]" : "";
      const state = item.state ? ` [estado=${item.state}]` : "";
      const decision = item.decision?.question
        ? `\nPregunta previa: ${item.decision.question} | opciones: ${item.decision.options.join(" | ")}`
        : "";

      return `${label}${image}${state}: ${item.text || item.displayText}${decision}`;
    })
    .join("\n");
}

function parseDecisionAnswer(text) {
  const value = String(text || "");
  if (!/RESPUESTA A PREGUNTA DECISIVA/i.test(value)) return null;

  const key = value.match(/^Variable:\s*(.+)$/im)?.[1]?.trim() || "";
  const question = value.match(/^Pregunta:\s*(.+)$/im)?.[1]?.trim() || "";
  const answer =
    value.match(/^Respuesta seleccionada por el cliente:\s*(.+)$/im)?.[1]?.trim() || "";

  if (!question || !answer) return null;
  return { key, question, answer };
}

function allTurns(currentMessage, conversation) {
  return [...conversation, { role: "user", text: currentMessage || "" }];
}

function collectResolvedFacts(currentMessage, conversation) {
  const facts = new Set();
  const turns = allTurns(currentMessage, conversation);

  for (const item of turns) {
    if (item.role !== "user" || !item.text) continue;

    const parsed = parseDecisionAnswer(item.text);
    if (parsed?.key) facts.add(parsed.key);

    const lower = item.text.toLowerCase();
    if (lower.includes("pregunta: ¿la playera será negra?")) facts.add("garment_black");
    if (lower.includes("pregunta: ¿dónde creaste la plantilla final?")) facts.add("template_source");
    if (lower.includes("pregunta: ¿revisaste la plantilla final")) facts.add("final_scan");
    if (lower.includes("pregunta: ¿cómo aparece el borde blanco?")) facts.add("white_border_pattern");
    if (lower.includes("pregunta: ¿qué tipo de diseño es?")) facts.add("design_type");
  }

  const userText = turns
    .filter((item) => item.role === "user")
    .map((item) => item.text || "")
    .join("\n");

  if (GARMENT_COLOR_REGEX.test(userText)) facts.add("garment_black");

  return facts;
}

function buildConfirmedFacts(currentMessage, conversation) {
  const facts = [];

  for (const item of allTurns(currentMessage, conversation)) {
    if (item.role !== "user" || !item.text) continue;

    const parsed = parseDecisionAnswer(item.text);
    if (parsed) {
      facts.push(`- ${parsed.question} → ${parsed.answer}`);
    }
  }

  return [...new Set(facts)];
}

function retrievalText(text) {
  const parsed = parseDecisionAnswer(text);
  if (parsed) return `${parsed.question} ${parsed.answer}`;
  return String(text || "");
}

function buildRetrievalQuery(message, conversation) {
  const recent = conversation
    .slice(-8)
    .map(
      (item) =>
        `${item.role === "assistant" ? "NOVAGE" : "CLIENTE"}: ${retrievalText(
          item.text || item.displayText
        ).slice(0, 700)}`
    )
    .join("\n");

  return [
    "Consulta actual:",
    retrievalText(message),
    recent ? `\nContexto reciente:\n${recent}` : "",
    "\nBusca solo información útil para resolver el problema de forma simple y correcta."
  ]
    .filter(Boolean)
    .join("\n")
    .slice(0, 12000);
}

async function searchVectorStore(vectorStoreId, query, maxResults = 7) {
  const response = await fetch(
    `https://api.openai.com/v1/vector_stores/${encodeURIComponent(vectorStoreId)}/search`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${process.env.OPENAI_API_KEY}`
      },
      body: JSON.stringify({
        query,
        max_num_results: maxResults,
        rewrite_query: true
      })
    }
  );

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data?.error?.message || "No se pudo consultar NOVAGE KNOWLEDGE.");
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

function formatSearchResults(results, maxChars = 14000) {
  const chunks = [];
  let used = 0;

  for (const result of results || []) {
    const text = resultText(result);
    if (!text) continue;

    const block = `FUENTE: ${result?.filename || "NOVAGE KNOWLEDGE"}\n${text}`;

    if (used + block.length > maxChars) break;
    chunks.push(block);
    used += block.length;
  }

  return chunks.join("\n\n---\n\n");
}

function slugKey(value) {
  return String(value || "")
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 50);
}

function buildCustomDecision(parsed, resolvedFacts) {
  const slug = slugKey(parsed?.custom_key);
  const question = String(parsed?.custom_question || "").trim().slice(0, 200);
  const message = String(parsed?.custom_message || "").trim().slice(0, 300);
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
    message: message || "Necesito un dato para darte la recomendación correcta.",
    question,
    options
  };
}

const GATE_INSTRUCTIONS = `
Eres el DECISION GATE de NOVAGE AI.

NO das soluciones. Solo decides si falta UN dato que realmente cambia la recomendación.

case_type:
- reference_only_art: el cliente NO tiene el diseño plano; solo tiene foto/mockup/captura de una prenda o producto con el estampado aplicado, con perspectiva, pliegues, objetos tapando o partes faltantes.
- detailed_black_background: diseño plano detallado/texturizado donde el negro forma parte importante de la composición.
- white_border_print: borde/halo blanco después de imprimir DTF.
- pixelated_design: pixelado, calidad, mejorar vs vectorizar.
- dtf_application: adherencia, planchado, lavado, film, polvo.
- sublimation
- vinyl
- file_preparation
- business
- other

missing_fact:
- garment_black
- template_source
- final_scan
- white_border_pattern
- design_type
- custom
- none

REGLAS:
1. Si case_type = reference_only_art, missing_fact = none. No interrogues. La solución será explicar que esa foto sirve como referencia, no como archivo de impresión.
2. Si el diseño plano es detallado, tiene fondo negro integrado y aún no se sabe el color de la prenda: garment_black.
3. Bordes blancos: pregunta primero template_source; después final_scan si fue app externa; después white_border_pattern si todavía cambia el diagnóstico.
4. Si mejorar vs vectorizar no se puede saber sin conocer el tipo de diseño y no hay imagen suficiente: design_type.
5. custom solo si existe UN dato que el cliente sabe y que cambia realmente la ruta.
6. No preguntes qué herramienta quiere usar.
7. No preguntes algo visible en la imagen.
8. Si el dato ya fue contestado, no lo vuelvas a preguntar.
9. Máximo una pregunta.

Si missing_fact != custom, los campos custom_* deben ir vacíos.
`;

const SOLUTION_INSTRUCTIONS = `
Eres NOVAGE AI, el asistente experto del Taller Digital de NOVAGE.

Tu prioridad es que cualquier principiante entienda QUÉ hacer, CUÁNDO hacerlo y POR QUÉ.
No intentes demostrar conocimiento técnico. Resuelve.

ESTILO:
- Respuestas concretas.
- Normalmente 60 a 140 palabras.
- Máximo 4 pasos.
- Frases cortas.
- Usa palabras comunes.
- Si usas un término técnico, explícalo en la misma frase.
- No repitas lo que el cliente ya dijo.
- No uses una estructura larga obligatoria de "Diagnóstico / Cómo comprobar / Si no se corrige" salvo que realmente haga falta.
- No des 3 soluciones si una es claramente la correcta.
- No llenes la respuesta de advertencias.

LÍMITES IMPORTANTES:
- Hay archivos que sí se pueden preparar y otros que primero necesitan un diseño correcto.
- Si una foto, mockup o captura NO contiene el arte plano completo, dilo claramente:
  "Esta imagen sirve como referencia, pero no como archivo directo para impresión."
- No mandes a un principiante a "reconstruir en Photoshop".
- Si no existe el arte original, recomienda conseguirlo o recrearlo con una IA de imágenes como ChatGPT o Gemini usando la foto como referencia.
- Solo DESPUÉS de tener un diseño plano, completo y limpio se recomienda entrar al flujo NOVAGE.
- El Mejorador con IA de NOVAGE sirve para mejorar/upscale de un DISEÑO EXISTENTE. No reconstruye partes ocultas, no elimina perspectiva y no inventa un diseño completo desde una foto de una playera.
- No comprometas al Taller con un resultado que sus herramientas no pueden producir.

CUANDO ALGO NO ES CONVENIENTE:
No des vueltas. Di:
- qué no conviene;
- por qué;
- qué necesita hacer antes.

EJEMPLO:
"Este diseño no se puede preparar bien solo quitando el fondo o creando semitonos. El negro forma parte de la composición. Para una playera de otro color necesitas una versión adaptada del diseño. Primero consigue o genera esa versión; después ya puedes usar NOVAGE para mejorarla, ajustar medidas y preparar la plantilla."

SI EL CLIENTE DICE QUE LA RESPUESTA ANTERIOR NO LE RESOLVIÓ:
- Empieza con una disculpa corta: "Lamento que no te haya resuelto."
- NO repitas la misma explicación.
- Identifica el límite real.
- Si el resultado que quiere no es posible con el archivo actual, dilo claramente.
- Da solo la alternativa útil.

CASOS CLAVE:
1. Diseño plano detallado + fondo negro + playera negra:
   - aprovechar negro de la prenda;
   - si la imagen es de baja calidad, Mejorador con IA;
   - después Semitonos Fáciles o Pro;
   - evitar imprimir una gran masa negra.

2. Diseño plano detallado + fondo negro + playera NO negra:
   - no eliminar el negro automáticamente;
   - si el negro define el diseño, necesita una versión adaptada para ese color de prenda;
   - si no tiene esa versión, puede generarla/recrearla con una IA de imágenes y luego trabajarla en NOVAGE.

3. Solo foto/mockup de una playera:
   - no usar Mejorador, Semitonos o Eliminador todavía;
   - conseguir/recrear primero el arte plano;
   - tools = [].

4. Mejorar vs vectorizar:
   - foto/anime/arte complejo -> Mejorador con IA.
   - logo/texto/gráfico plano/pocos colores -> Vectorizador NOVAGE.

5. Bordes blancos:
   - no recomendar Contraer Bordes automáticamente;
   - si el blanco está corrido hacia un lado y el archivo final estaba limpio, sospechar impresión/desfase de tinta blanca;
   - si rodea todo el borde, revisar archivo, semitransparencias o residuos.

HERRAMIENTAS:
Devuelve en "tools" solo herramientas NOVAGE que el usuario YA puede usar en ese momento.
Si primero necesita conseguir, recrear o corregir el arte fuera del Taller, devuelve [].
Máximo 3 herramientas.
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

async function runDecisionGate({
  message,
  image,
  imageFromContext,
  sharedContext,
  resolvedFacts
}) {
  const content = [
    {
      type: "input_text",
      text: [
        `Consulta actual:\n${message || "Analiza la imagen adjunta."}`,
        imageFromContext
          ? "\n[La imagen adjunta es la misma que el cliente compartió antes.]"
          : "",
        `\nVariables ya resueltas: ${[...resolvedFacts].join(", ") || "ninguna"}`,
        sharedContext
      ].join("\n")
    }
  ];

  if (image) {
    content.push({
      type: "input_image",
      image_url: image,
      detail: "high"
    });
  }

  const data = await callOpenAI({
    model: "gpt-6-luna",
    instructions: GATE_INSTRUCTIONS,
    input: [{ role: "user", content }],
    reasoning: { effort: "low" },
    max_output_tokens: 1800,
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
                "reference_only_art",
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
    prompt_cache_key: "novage-gate-v5"
  });

  return parseJsonOutput(data, {
    case_type: "other",
    missing_fact: "none",
    custom_key: "",
    custom_message: "",
    custom_question: "",
    custom_options: [],
    confidence: "low",
    reason: ""
  });
}

async function runSolution({
  message,
  image,
  imageFromContext,
  sharedContext,
  gateInfo,
  dissatisfaction
}) {
  const content = [
    {
      type: "input_text",
      text: [
        `Consulta actual:\n${message || "Analiza la imagen adjunta."}`,
        imageFromContext
          ? "\n[La imagen adjunta es la misma que el cliente compartió antes. Analízala.]"
          : "",
        `\nCLASIFICACIÓN DEL CASO: ${gateInfo.caseType}`,
        gateInfo.reason ? `\nMotivo: ${gateInfo.reason}` : "",
        dissatisfaction
          ? "\nIMPORTANTE: el cliente dijo que la respuesta anterior no le resolvió. No repitas la misma ruta. Sé breve, reconoce el límite real y da la alternativa útil."
          : "",
        sharedContext
      ].join("\n")
    }
  ];

  if (image) {
    content.push({
      type: "input_image",
      image_url: image,
      detail: "high"
    });
  }

  const data = await callOpenAI({
    model: "gpt-6-luna",
    instructions: SOLUTION_INSTRUCTIONS,
    input: [{ role: "user", content }],
    reasoning: { effort: "medium" },
    max_output_tokens: 2600,
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
    store: false,
    prompt_cache_key: "novage-solution-v5"
  });

  const parsed = parseJsonOutput(data, {
    message: "No pude terminar la respuesta. Intenta enviarla de nuevo.",
    tools: []
  });

  const tools = Array.isArray(parsed.tools)
    ? [...new Set(parsed.tools.filter((name) => TOOL_NAMES.includes(name)))].slice(0, 3)
    : [];

  return {
    message:
      typeof parsed.message === "string" && parsed.message.trim()
        ? parsed.message.trim()
        : "No pude generar una respuesta.",
    tools
  };
}

export default async function handler(req, res) {
  setCors(req, res);

  const origin = getOrigin(req);
  const host = getHost(req);
  const secFetchSite = String(req.headers["sec-fetch-site"] || "").toLowerCase();

  if (!isAllowedOrigin(origin)) {
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

  try {
    verifySessionToken(req);

    if (!process.env.OPENAI_API_KEY) {
      return res.status(500).json({
        error: "OPENAI_API_KEY no está configurada."
      });
    }

    if (!process.env.NOVAGE_VECTOR_STORE_ID) {
      return res.status(500).json({
        error: "NOVAGE_VECTOR_STORE_ID no está configurado."
      });
    }

    const limit = checkRateLimit(getClientIp(req));

    if (!limit.allowed) {
      return res.status(429).json({
        error: "Has alcanzado el límite temporal. Intenta nuevamente más tarde."
      });
    }

    const {
      mensaje,
      imagen: imagenRaw,
      imagenContexto,
      conversationContext
    } = req.body || {};

    const message =
      typeof mensaje === "string" ? mensaje.trim().slice(0, 6000) : "";

    const image =
      typeof imagenRaw === "string" && imagenRaw.startsWith("data:image/")
        ? imagenRaw
        : null;

    const imageFromContext = Boolean(image && imagenContexto === true);

    if (!message && !image) {
      return res.status(400).json({
        error: "Escribe un mensaje o sube una imagen."
      });
    }

    const conversation = cleanConversationContext(conversationContext);
    const resolvedFacts = collectResolvedFacts(message, conversation);
    const confirmedFacts = buildConfirmedFacts(message, conversation);

    const retrievalQuery = buildRetrievalQuery(
      message || "Analiza el archivo adjunto.",
      conversation
    );

    let knowledgeResults = [];

    try {
      knowledgeResults = await searchVectorStore(
        process.env.NOVAGE_VECTOR_STORE_ID,
        retrievalQuery,
        KNOWLEDGE_RESULTS
      );
    } catch (error) {
      console.error("NOVAGE knowledge search error:", error);
    }

    const officialContext = formatSearchResults(knowledgeResults);

    const sharedContext = `
HECHOS CONFIRMADOS:
${confirmedFacts.length ? confirmedFacts.join("\n") : "Ninguno adicional."}

CONOCIMIENTO OFICIAL NOVAGE:
${officialContext || "No se recuperó información específica. No inventes funciones de NOVAGE."}

CONVERSACIÓN ACTUAL:
${conversationAsText(conversation) || "Sin contexto previo."}
`;

    let gate;

    try {
      gate = await runDecisionGate({
        message,
        image,
        imageFromContext,
        sharedContext,
        resolvedFacts
      });
    } catch (error) {
      console.error("NOVAGE gate error:", error);
      gate = {
        case_type: "other",
        missing_fact: "none",
        confidence: "low",
        reason: ""
      };
    }

    let missingFact = gate?.missing_fact || "none";

    if (resolvedFacts.has(missingFact)) {
      missingFact = "none";
    }

    if (
      gate?.case_type === "reference_only_art"
    ) {
      missingFact = "none";
    }

    let decision = null;

    if (missingFact === "custom") {
      decision = buildCustomDecision(gate, resolvedFacts);
    } else if (missingFact !== "none" && DECISIONS[missingFact]) {
      decision = {
        key: missingFact,
        ...DECISIONS[missingFact]
      };
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
        remaining: limit.remaining
      });
    }

    const solution = await runSolution({
      message,
      image,
      imageFromContext,
      sharedContext,
      gateInfo: {
        caseType: gate?.case_type || "other",
        reason: gate?.reason || ""
      },
      dissatisfaction: DISSATISFACTION_REGEX.test(message)
    });

    return res.status(200).json({
      respuesta: solution.message,
      estado: "solution",
      decision: {
        key: "",
        question: "",
        options: []
      },
      tools: solution.tools,
      remaining: limit.remaining
    });
  } catch (error) {
    console.error("NOVAGE AI Error:", error);

    return res.status(500).json({
      error: "Ocurrió un error al procesar la solicitud.",
      details: error?.message || "Error desconocido"
    });
  }
}
