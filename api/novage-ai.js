export default async function handler(req, res) {
  // Permitir únicamente POST
  if (req.method !== "POST") {
    return res.status(405).json({
      error: "Método no permitido",
    });
  }

  try {
    const { mensaje } = req.body || {};

    if (!mensaje || !mensaje.trim()) {
      return res.status(400).json({
        error: "Escribe un mensaje.",
      });
    }

    const response = await fetch("https://api.openai.com/v1/responses", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
      },
      body: JSON.stringify({
        model: "gpt-6-luna",

        instructions: `
Eres NOVAGE AI, el asistente especializado de NOVAGE.

Ayudas principalmente con:
- Impresión DTF.
- Preparación de diseños para impresión.
- Calidad y resolución de imágenes.
- Eliminación de fondos.
- Vectorización.
- Semitonos.
- Semitransparencias.
- Medidas para impresión.
- Mockups.
- Herramientas de NOVAGE.

Responde siempre en español, salvo que el usuario pida otro idioma.

Tus respuestas deben ser:
- Claras.
- Directas.
- Fáciles de entender.
- Prácticas.
- No demasiado largas.

Cuando sea apropiado, recomienda herramientas de NOVAGE.

No inventes funciones, productos o características de NOVAGE que no conozcas.
        `,

        input: mensaje,
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      console.error("OpenAI API Error:", data);

      return res.status(response.status).json({
        error: "OpenAI no pudo generar una respuesta.",
        details: data?.error?.message || "Error desconocido",
      });
    }

    const respuesta =
      data.output
        ?.flatMap((item) => item.content || [])
        ?.find((item) => item.type === "output_text")
        ?.text || "No se pudo obtener una respuesta.";

    return res.status(200).json({
      respuesta,
    });
  } catch (error) {
    console.error("NOVAGE AI Error:", error);

    return res.status(500).json({
      error: "Ocurrió un error interno.",
    });
  }
}
