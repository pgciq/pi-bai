// B.AI provider — https://docs.b.ai/llmservice/api/
//
// B.AI is a unified LLM gateway compatible with three protocols:
//   • OpenAI Chat Completions  → POST /v1/chat/completions
//   • OpenAI Responses         → POST /v1/responses   ← used here for chat
//   • Anthropic Messages        → POST /v1/messages
//
// Chat is routed through the OpenAI Responses API because it exposes native
// reasoning controls (effort + summary). We delegate to pi-ai's built-in
// openai-responses core, which maps Responses SSE events → pi's assistant event
// stream (including thinking_start/delta/end) and assembles the `reasoning`
// block from `reasoning.effort` / `reasoning.summary`.
//
// Auth: BAI_API_KEY, sent as `Authorization: Bearer <key>`.
// Model catalog: GET /v1/models (OpenAI-compatible) → refreshModels.
//
// Image generation (OpenAI-compatible POST /v1/images/generations) is wired as a
// native streamer and dispatched from streamBai via the `baiImageModel` flag —
// the same pattern pi-cloudflare-workers-ai uses. pi's extension API has no
// separate image-registration surface, so image models live in the same provider
// and are flagged for the streamer (the live /v1/models catalog does not include
// image models, so they are always merged back in).

import { execFile } from "node:child_process";
import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { createAssistantMessageEventStream } from "@earendil-works/pi-ai";
import { Image, Markdown } from "@earendil-works/pi-tui";
import { getMarkdownTheme } from "@earendil-works/pi-coding-agent";

const API_BASE = "https://api.b.ai/v1";
const API_KEY_ENV = "BAI_API_KEY";

// OpenAI-compatible image generation endpoint. B.AI documents GPT-Image-2 as an
// "OpenAI image generation ... model", so we assume the standard
// POST /v1/images/generations shape:
//   request : { model, prompt, n, size, response_format }
//   response: { data: [ { b64_json } | { url } ] }
// If your B.AI plan exposes a different path, change IMAGE_GEN_PATH.
const IMAGE_GEN_PATH = "/images/generations";
const IMAGE_DIR = join(process.cwd(), ".pi", "generated-images");

// pi-ai exposes openAIResponsesApi via a lazy wrapper (preferred). Resolve it so
// the extension loads on any pi-ai build; fall back to a throwing stub only if
// the core is entirely absent.
const openAIResponsesApi = await (async () => {
  try {
    return (await import("@earendil-works/pi-ai/api/openai-responses.lazy")).openAIResponsesApi;
  } catch {
    return () => ({
      streamSimple: () => {
        throw new Error("pi-ai openai-responses core is unavailable in this environment.");
      },
    });
  }
})();

// Chat-completions core — used for models that are NOT supported on the Responses
// API (e.g. the limited-time-free deepseek-v4-flash / hy3 / mimo-v2.5 families,
// which B.AI rejects on /v1/responses with model_not_supported_on_endpoint).
const openAICompletionsApi = await (async () => {
  try {
    return (await import("@earendil-works/pi-ai/api/openai-completions.lazy")).openAICompletionsApi;
  } catch {
    return () => ({
      streamSimple: () => {
        throw new Error("pi-ai openai-completions core is unavailable in this environment.");
      },
    });
  }
})();

// Reasoning-capable families (per B.AI docs: GPT-5.x, Claude Opus 5, Gemini 3.6,
// DeepSeek v4, GLM-5.3, Kimi, Grok, MiMo, MiniMax, Hunyuan all support
// reasoning). Lighter tiers are left non-reasoning to avoid sending the
// `reasoning` param to models that may not accept it.
const REASONING_IDS = new Set([
  "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna",
  "gpt-5.5", "gpt-5.4", "gpt-5.4-pro",
  "claude-opus-5", "gemini-3.6-flash", "deepseek-v4-pro",
  "glm-5.3-flash", "kimi-k3", "qwen3.8-flash",
  "mimo-v2.5-pro", "minimax-m3",
]);

// Models that B.AI only exposes on /v1/chat/completions (they 400 on
// /v1/responses with model_not_supported_on_endpoint). The limited-time-free
// models fall here. These still reason natively (they emit reasoning tokens on
// chat/completions), so we route them through the openai-completions core.
const CHAT_ONLY_IDS = new Set([
  "deepseek-v4-flash",
  "deepseek-v4-flash-vision-exp",
  "hy3",
  "mimo-v2.5",
]);

// Maps pi thinking levels → B.AI reasoning.effort values.
const THINKING_LEVEL_MAP: Record<string, string> = {
  off: "none",
  minimal: "low",
  low: "low",
  medium: "medium",
  high: "high",
  xhigh: "xhigh",
  max: "max",
};

// Best-effort seed of chat-model IDs (doc URL slug → API id, version dots restored).
const CHAT_SEED_IDS = [
  "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna",
  "gpt-5.5", "gpt-5.5-instant", "gpt-5.4", "gpt-5.4-pro",
  "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.2",
  "gpt-5-mini", "gpt-5-nano",
  "claude-opus-5", "gemini-3.6-flash", "deepseek-v4-pro",
  "glm-5.3-flash", "kimi-k3", "qwen3.8-flash",
  "mimo-v2.5-pro", "minimax-m3", "hy3",
  // Limited-time-free models — chat/completions only (rejected on /v1/responses).
  "deepseek-v4-flash", "deepseek-v4-flash-vision-exp", "mimo-v2.5",
];

// Image-generation models. IDs are best-effort from the B.AI docs slugs
// (gpt-image-2, grok-imagine-image-2.0). VERIFY before relying on them:
// live tests returned `model_not_found` for these IDs and the image API
// reference was rate-limited at implementation time, so the exact image model
// IDs could not be confirmed. The /v1/images/generations PATH is confirmed to
// exist (the API reaches model resolution); only the IDs are uncertain. The
// live /v1/models catalog never includes image models, so they are seeded here.
const IMAGE_SEED_IDS = ["gpt-image-2", "grok-imagine-image-2.0"];

const DEFAULT_CONTEXT = 200000;
const DEFAULT_MAX_TOKENS = 64000;
const ZERO_COST = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };

function chatModel(id: string) {
  const chatOnly = CHAT_ONLY_IDS.has(id);
  const reasoning = !chatOnly && REASONING_IDS.has(id);
  return {
    id,
    name: id,
    reasoning,
    input: ["text"],
    cost: { ...ZERO_COST },
    contextWindow: DEFAULT_CONTEXT,
    maxTokens: DEFAULT_MAX_TOKENS,
    // Private metadata (pi ignores unknown fields); consumed by streamBai.
    ...(chatOnly ? { baiChatOnly: true } : {}),
    ...(reasoning ? { thinkingLevelMap: THINKING_LEVEL_MAP } : {}),
  };
}

function imageModel(id: string) {
  return {
    id,
    name: id,
    reasoning: false,
    input: ["text"],
    cost: { ...ZERO_COST },
    contextWindow: DEFAULT_CONTEXT,
    maxTokens: DEFAULT_MAX_TOKENS,
    // Private metadata (pi ignores unknown fields); consumed by streamBai.
    baiImageModel: true,
    baiImageSize: "1024x1024",
  };
}

// ---------------------------------------------------------------------------
// Model catalog
// ---------------------------------------------------------------------------
async function fetchModels(baseUrl: string, signal?: AbortSignal) {
  const apiKey = process.env[API_KEY_ENV];
  const headers: Record<string, string> = {};
  if (apiKey) headers["Authorization"] = `Bearer ${apiKey}`;

  const res = await fetch(`${baseUrl}/models`, { headers, redirect: "follow", signal });
  if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);

  const payload = await res.json();
  // OpenAI /v1/models returns { data: [{ id, ... }] }
  const data = Array.isArray(payload?.data)
    ? payload.data
    : Array.isArray(payload)
      ? payload
      : [];
  return data.filter((m: any) => m && m.id).map((m: any) => chatModel(String(m.id)));
}

// ---------------------------------------------------------------------------
// Image generation (native /v1/images/generations)
// ---------------------------------------------------------------------------
let appendBaiImage: ((image: { path: string; mimeType: string }) => void) | null = null;

function makeOutput(model: any) {
  return {
    role: "assistant",
    content: [] as any[],
    api: model.api,
    provider: model.provider,
    model: model.id,
    usage: {
      input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
    stopReason: "stop",
    timestamp: Date.now(),
  };
}

function fileLink(p: string, label = p): string {
  return `[${label}](${pathToFileURL(String(p)).href})`;
}

function latestTextPrompt(context: any): string {
  const messages = Array.isArray(context?.messages) ? context.messages : [];
  const user = [...messages].reverse().find((m: any) => m?.role === "user");
  if (!user) return "";
  if (typeof user.content === "string") return user.content;
  return (user.content ?? [])
    .filter((p: any) => p?.type === "text")
    .map((p: any) => p.text ?? "")
    .join("\n");
}

async function saveGeneratedImage(image: { b64_json?: string; url?: string }, modelId: string) {
  if (!existsSync(IMAGE_DIR)) mkdirSync(IMAGE_DIR, { recursive: true });
  const mimeType = image?.mime_type ?? image?.mimeType ?? "image/png";
  const ext = mimeType.includes("jpeg") ? "jpg" : mimeType.includes("webp") ? "webp" : "png";
  const safe = String(modelId).replace(/[\/.]/g, "-");
  const filePath = join(IMAGE_DIR, `bai-${safe}-${Date.now()}.${ext}`);

  if (image?.b64_json) {
    writeFileSync(filePath, Buffer.from(image.b64_json, "base64"));
  } else if (image?.url) {
    const r = await fetch(image.url);
    if (!r.ok) throw new Error(`Failed to download generated image: HTTP ${r.status}`);
    writeFileSync(filePath, Buffer.from(await r.arrayBuffer()));
  } else {
    throw new Error("Image API returned neither url nor b64_json");
  }
  return { filePath, mimeType };
}

function streamImageGeneration(model: any, context: any, options: any) {
  const stream = createAssistantMessageEventStream();
  const output = makeOutput(model);
  (async () => {
    try {
      stream.push({ type: "start", partial: output });
      const prompt = latestTextPrompt(context);
      if (!prompt) throw new Error("Image generation requires a text prompt");
      const apiKey = process.env[API_KEY_ENV];
      if (!apiKey) throw new Error(`${API_KEY_ENV} env var is required for image generation`);

      const response = await fetch(`${API_BASE}${IMAGE_GEN_PATH}`, {
        method: "POST",
        headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
        body: JSON.stringify({
          model: model.id,
          prompt,
          n: 1,
          size: model.baiImageSize || "1024x1024",
          response_format: "b64_json",
        }),
        signal: options?.signal,
      });

      const contentType = response.headers.get("content-type") ?? "";
      let payload: any;
      if (contentType.includes("application/json")) {
        payload = await response.json();
        if (!response.ok) throw new Error(payload?.error?.message ?? `B.AI image API HTTP ${response.status}`);
      } else {
        if (!response.ok) throw new Error(`B.AI image API HTTP ${response.status}`);
        // Some gateways return raw image bytes instead of JSON.
        payload = { data: [{ b64_json: Buffer.from(await response.arrayBuffer()).toString("base64") }] };
      }

      const item = payload?.data?.[0];
      if (!item || (!item.b64_json && !item.url)) {
        throw new Error("B.AI image API returned no image data");
      }

      const saved = await saveGeneratedImage(item, model.id);
      appendBaiImage?.({ path: saved.filePath, mimeType: saved.mimeType });
      const text = `Generated image saved to: ${fileLink(saved.filePath)}`;
      output.content.push({ type: "text", text });
      stream.push({ type: "text_start", contentIndex: 0, partial: output });
      stream.push({ type: "text_delta", contentIndex: 0, delta: text, partial: output });
      stream.push({ type: "text_end", contentIndex: 0, content: text, partial: output });
      output.stopReason = "stop";
      stream.push({ type: "done", reason: "stop", message: output });
      stream.end();
    } catch (error) {
      output.stopReason = options?.signal?.aborted ? "aborted" : "error";
      output.errorMessage = error instanceof Error ? error.message : String(error);
      stream.push({ type: "error", reason: output.stopReason, error: output });
      stream.end();
    }
  })();
  return stream;
}

// ---------------------------------------------------------------------------
// Chat (OpenAI Responses API) — delegated to pi-ai's openai-responses core
// ---------------------------------------------------------------------------
function streamResponsesChat(model: any, context: any, options: any) {
  const resolved = { ...model, baseUrl: API_BASE };
  return openAIResponsesApi().streamSimple(resolved, context, {
    ...options,
    apiKey: process.env[API_KEY_ENV],
  });
}

function streamBai(model: any, context: any, options: any) {
  if (model.baiImageModel) return streamImageGeneration(model, context, options);
  // Chat-only models (e.g. the limited-time-free deepseek-v4-flash / hy3 /
  // mimo-v2.5 families) are rejected on /v1/responses; route them to the
  // openai-completions core. All other models use the Responses API, which
  // carries native reasoning effort / summary controls.
  const apiKey = process.env[API_KEY_ENV];
  if (model.baiChatOnly) {
    const resolved = { ...model, baseUrl: API_BASE, api: "openai-completions" };
    return openAICompletionsApi().streamSimple(resolved, context, { ...options, apiKey });
  }
  return streamResponsesChat(model, context, options);
}

// ---------------------------------------------------------------------------
// Commands: /bai-models, /bai-docs
// ---------------------------------------------------------------------------
const DOCS_URL = "https://docs.b.ai/llmservice/api/";

function baiModels(ctx: any) {
  const all = ctx?.modelRegistry?.getAvailable?.() ?? [];
  return all.filter((m: any) => m.provider === "bai");
}

function modelType(m: any): "image" | "chat" {
  return m.baiImageModel ? "image" : "chat";
}

function fmtSize(n?: number): string {
  if (!n) return "—";
  if (n >= 1_048_576) return `${Math.round(n / 1_048_576)}M`;
  if (n >= 1024) return `${Math.round(n / 1024)}K`;
  return String(n);
}

function buildModelsMarkdown(models: any[]): string {
  const head = [
    "# B.AI models",
    "",
    `_${models.length} model(s) — catalog from \`GET /v1/models\` (image models always included)_`,
  ];
  if (models.length === 0) {
    return [...head, "", "_No B.AI models available — set `BAI_API_KEY` and restart pi._"].join("\n");
  }
  const rows = models
    .map(
      (m) =>
        `| \`${m.id}\` | ${modelType(m)} | ${m.reasoning ? "✓" : "—"} | ${fmtSize(m.contextWindow)} | ${fmtSize(m.maxTokens)} |`,
    )
    .join("\n");
  return [
    ...head,
    "",
    "| Model | Type | Reasoning | Context | Max Out |",
    "|---|---|:---:|:---:|---:|",
    rows,
    "",
    "_Context windows / max output are conservative defaults; B.AI does not expose them via `/v1/models`._",
  ].join("\n");
}

function openBrowser(url: string) {
  const cmd = process.platform === "darwin" ? "open" : process.platform === "win32" ? "cmd" : "xdg-open";
  const args = process.platform === "win32" ? ["/c", "start", "", url] : [url];
  try {
    execFile(cmd, args, () => {});
  } catch {
    /* best-effort */
  }
}

function notifyOrPrint(ctx: any, message: string, level: "info" | "warning" = "warning") {
  if (ctx?.hasUI) ctx.ui.notify(message, level);
  else console.log(message);
}

function registerBaiCommands(pi: any) {
  pi.registerCommand("bai-models", {
    description: "List B.AI models available to pi (chat via Responses API + image generation).",
    handler: async (_args: string, ctx: any) => {
      const markdown = buildModelsMarkdown(baiModels(ctx));
      if (ctx?.mode === "tui") pi.appendEntry("bai-models", { markdown });
      else notifyOrPrint(ctx, markdown, "info");
    },
  });
  pi.registerEntryRenderer?.("bai-models", (entry: any) =>
    new Markdown(entry.data?.markdown ?? "", 1, 0, getMarkdownTheme()),
  );

  pi.registerCommand("bai-docs", {
    description: "Open the B.AI API documentation in your browser.",
    handler: async (_args: string, ctx: any) => {
      openBrowser(DOCS_URL);
      const note = `Opened B.AI API docs: ${DOCS_URL}`;
      if (ctx?.mode === "tui") pi.appendEntry("bai-docs", { markdown: `# B.AI docs\n\n${note}` });
      else notifyOrPrint(ctx, note, "info");
    },
  });
  pi.registerEntryRenderer?.("bai-docs", (entry: any) =>
    new Markdown(entry.data?.markdown ?? "", 1, 0, getMarkdownTheme()),
  );
}

// ---------------------------------------------------------------------------
// Extension entry point (synchronous — no network on startup)
// ---------------------------------------------------------------------------
export default function (pi: any) {
  appendBaiImage = (image) => pi.appendEntry("bai-generated-image", image);
  pi.registerEntryRenderer?.("bai-generated-image", (entry: any, _options: any, theme: any) => {
    const image = entry.data ?? {};
    // pi passes an entry-renderer theme that lacks fallbackColor(), which
    // Image.render calls. Wrap it so inline previews render and never throw.
    const imageTheme =
      theme && typeof theme.fallbackColor === "function"
        ? theme
        : { fallbackColor: (s: string) => (theme && theme.fg ? theme.fg("toolOutput", s) : s) };
    try {
      const data = readFileSync(image.path).toString("base64");
      return new Image(data, image.mimeType || "image/png", imageTheme, { maxWidthCells: 80, maxHeightCells: 30 });
    } catch {
      return new Markdown(`Generated image unavailable: ${fileLink(image.path ?? "unknown path")}`, 1, 0, theme);
    }
  });

  registerBaiCommands(pi);

  pi.registerProvider("bai", {
    name: "B.AI",
    baseUrl: API_BASE,
    // Keep this as an env reference even when the variable is absent, so pi can
    // mark the provider as unconfigured instead of using a placeholder key.
    apiKey: `$${API_KEY_ENV}`,
    api: "openai-responses",
    streamSimple: streamBai,
    models: [...CHAT_SEED_IDS.map(chatModel), ...IMAGE_SEED_IDS.map(imageModel)],

    async refreshModels({ signal, stored, publish, allowNetwork }: any) {
      const cached = Array.isArray(stored?.models) ? stored.models : undefined;
      const seed = [...CHAT_SEED_IDS.map(chatModel), ...IMAGE_SEED_IDS.map(imageModel)];

      const mergeImageModels = (models: any[]) => {
        const seen = new Set(models.map((m) => m.id));
        for (const im of IMAGE_SEED_IDS.map(imageModel)) {
          if (!seen.has(im.id)) models.push(im);
        }
        return models;
      };

      if (allowNetwork === false || signal?.aborted) {
        return cached?.length ? mergeImageModels(cached) : seed;
      }

      try {
        const fetched = await fetchModels(API_BASE, signal);
        if (fetched.length > 0) {
          const merged = mergeImageModels(fetched);
          await publish({ persist: { provider: "bai", models: merged } });
          return merged;
        }
      } catch {
        /* fall through to cache / seed */
      }

      return cached?.length ? mergeImageModels(cached) : seed;
    },
  });
}
