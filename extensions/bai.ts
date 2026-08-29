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

// lazyStream builds an AssistantMessageEventStream from an async iterable — used
// to wrap the Responses attempt so we can transparently retry on Chat
// Completions when B.AI reports model_not_supported_on_endpoint.
const lazyStream = await (async () => {
  try {
    return (await import("@earendil-works/pi-ai/api/lazy")).lazyStream;
  } catch {
    return null;
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
  // glm-5.3-flash is a reasoning model but is also rejected on /v1/responses.
  "glm-5.3-flash",
]);

// Models confirmed usable on the FREE tier (a request to /v1/chat/completions
// returns HTTP 200 without a deposit). Everything else currently returns
// 403 access_denied ("Deposit required to unlock premium models") or 400
// insufficient_user_quota. B.AI does NOT expose this in /v1/models, so this is a
// curated, point-in-time list — re-probe to refresh. The /bai-models command
// badges these **FREE**. Determined by probing all 44 catalog models:
//   FREE      : deepseek-v4-flash, deepseek-v4-flash-vision-exp, hy3, mimo-v2.5,
//               glm-5.3-flash, qwen3.8-flash
//   PREMIUM   : gpt-5.*, claude-*, gemini-*, glm-5.1/5.2, kimi-*, deepseek-v4-pro, ...
//   QUOTA     : minimax-m2.7, qwen3.8-27b, mimo-v2.5-pro (400 insufficient_user_quota)
const FREE_IDS = new Set([
  "deepseek-v4-flash",
  "deepseek-v4-flash-vision-exp",
  "hy3",
  "mimo-v2.5",
  "glm-5.3-flash",
  "qwen3.8-flash",
]);

// ---------------------------------------------------------------------------
// Model access status — learned from live API responses
// ---------------------------------------------------------------------------
// B.AI changes pricing over time (free <-> charge). Instead of trusting only the
// static FREE_IDS seed, we record each model's real access tier from actual API
// responses and can refresh everything on demand via /bai-free:
//   200                    -> "free"
//   403 access_denied      -> "premium"  (deposit required)
//   400 insufficient_quota -> "quota"
// Observed status overrides the seed, so the /bai-models badge self-corrects the
// next time you use a model that flipped. Unknown models fall back to FREE_IDS.
export const modelStatus = new Map(); // id -> "free" | "premium" | "quota"

function recordStatus(id: string, status: string) { modelStatus.set(id, status); }

function recordStatusFromError(id: string, ev: any) {
  const msg = [
    ev?.errorMessage, ev?.message, ev?.error?.message, ev?.error?.code,
    JSON.stringify(ev?.error ?? {}),
  ].filter(Boolean).join(" ").toLowerCase();
  if (msg.includes("access_denied") || msg.includes("deposit")) recordStatus(id, "premium");
  else if (msg.includes("insufficient_user_quota") || msg.includes("quota")) recordStatus(id, "quota");
}

// Access tier for display: observed status wins; otherwise fall back to the seed.
function accessTier(id: string): string | null {
  const observed = modelStatus.get(id);
  if (observed) return observed;
  return FREE_IDS.has(id) ? "free" : null;
}

// Wrap a chat stream so we learn the model's access tier from its outcome.
// pi's lazyStream(model, setup) calls setup() with no args and expects a Promise
// resolving to an async-iterable, so the observer is returned from an async arrow.
export function observe(modelId: string, stream: any) {
  if (!lazyStream) return stream;
  return lazyStream({ id: modelId, provider: "bai" } as any, async () => {
    async function* gen() {
      let err: any = null;
      for await (const ev of stream) {
        if (ev?.type === "error") err = ev;
        else if (ev?.type === "done" && ev.reason !== "error") recordStatus(modelId, "free");
        if (err) recordStatusFromError(modelId, err);
        yield ev;
      }
    }
    return gen();
  });
}

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

// ---------------------------------------------------------------------------
// Model capabilities (context window / max output / input modalities)
// ---------------------------------------------------------------------------
// B.AI's `GET /v1/models` returns only { id, object, created } — it does NOT expose
// context windows, max output, or modalities. The table below is sourced from each
// model's public specifications (best-effort; B.AI-specific version suffixes such as
// gpt-5.6-sol share their family's limits). familyCaps() infers a sensible default
// for any model not explicitly listed (e.g. one added to the live catalog later).
interface Caps { contextWindow: number; maxTokens: number; input: string[]; }
const MODEL_CAPS: Record<string, Caps> = {
  // OpenAI GPT-5.x
  "gpt-5.6-sol": { contextWindow: 400000, maxTokens: 128000, input: ["text", "image"] },
  "gpt-5.6-terra": { contextWindow: 400000, maxTokens: 128000, input: ["text", "image"] },
  "gpt-5.6-luna": { contextWindow: 400000, maxTokens: 128000, input: ["text", "image"] },
  "gpt-5.5": { contextWindow: 400000, maxTokens: 128000, input: ["text", "image"] },
  "gpt-5.5-instant": { contextWindow: 400000, maxTokens: 128000, input: ["text", "image"] },
  "gpt-5.4": { contextWindow: 400000, maxTokens: 128000, input: ["text", "image"] },
  "gpt-5.2": { contextWindow: 400000, maxTokens: 128000, input: ["text", "image"] },
  "gpt-5.4-pro": { contextWindow: 400000, maxTokens: 128000, input: ["text", "image"] },
  "gpt-5.4-mini": { contextWindow: 400000, maxTokens: 128000, input: ["text", "image"] },
  "gpt-5-mini": { contextWindow: 400000, maxTokens: 64000, input: ["text", "image"] },
  "gpt-5.4-nano": { contextWindow: 400000, maxTokens: 16000, input: ["text", "image"] },
  "gpt-5-nano": { contextWindow: 400000, maxTokens: 16000, input: ["text", "image"] },
  // Anthropic Claude 5.x
  "claude-opus-5": { contextWindow: 200000, maxTokens: 64000, input: ["text", "image"] },
  "claude-fable-5": { contextWindow: 200000, maxTokens: 64000, input: ["text", "image"] },
  "claude-opus-4.8": { contextWindow: 200000, maxTokens: 64000, input: ["text", "image"] },
  "claude-opus-4.7": { contextWindow: 200000, maxTokens: 64000, input: ["text", "image"] },
  "claude-opus-4.6": { contextWindow: 200000, maxTokens: 64000, input: ["text", "image"] },
  "claude-opus-4.5": { contextWindow: 200000, maxTokens: 64000, input: ["text", "image"] },
  "claude-sonnet-5": { contextWindow: 200000, maxTokens: 64000, input: ["text", "image"] },
  "claude-sonnet-4.6": { contextWindow: 200000, maxTokens: 64000, input: ["text", "image"] },
  "claude-sonnet-4.5": { contextWindow: 200000, maxTokens: 64000, input: ["text", "image"] },
  "claude-haiku-4.5": { contextWindow: 200000, maxTokens: 64000, input: ["text", "image"] },
  // Google Gemini 3.x
  "gemini-3.1-pro": { contextWindow: 1000000, maxTokens: 64000, input: ["text", "image"] },
  "gemini-3-flash": { contextWindow: 1000000, maxTokens: 64000, input: ["text", "image"] },
  "gemini-3.5-flash-lite": { contextWindow: 1000000, maxTokens: 64000, input: ["text", "image"] },
  "gemini-3.6-flash": { contextWindow: 1000000, maxTokens: 64000, input: ["text", "image"] },
  "gemini-3.5-flash": { contextWindow: 1000000, maxTokens: 64000, input: ["text", "image"] },
  // DeepSeek v4
  "deepseek-v4-flash": { contextWindow: 128000, maxTokens: 32000, input: ["text"] },
  "deepseek-v4-flash-vision-exp": { contextWindow: 128000, maxTokens: 32000, input: ["text", "image"] },
  "deepseek-v4-pro": { contextWindow: 128000, maxTokens: 32000, input: ["text", "image"] },
  // Zhipu GLM-5.x
  "glm-5.1": { contextWindow: 128000, maxTokens: 32000, input: ["text", "image"] },
  "glm-5.2": { contextWindow: 128000, maxTokens: 32000, input: ["text", "image"] },
  "glm-5.3": { contextWindow: 128000, maxTokens: 32000, input: ["text", "image"] },
  "glm-5.3-flash": { contextWindow: 128000, maxTokens: 32000, input: ["text", "image"] },
  // Moonshot Kimi
  "kimi-k2.6": { contextWindow: 256000, maxTokens: 32000, input: ["text"] },
  "kimi-k3": { contextWindow: 256000, maxTokens: 32000, input: ["text"] },
  // Alibaba Qwen3.8
  "qwen3.8-flash": { contextWindow: 256000, maxTokens: 32000, input: ["text", "image"] },
  "qwen3.8-max": { contextWindow: 256000, maxTokens: 32000, input: ["text", "image"] },
  "qwen3.8-27b": { contextWindow: 256000, maxTokens: 32000, input: ["text", "image"] },
  // MiniMax
  "minimax-m3": { contextWindow: 256000, maxTokens: 32000, input: ["text"] },
  "minimax-m2.7": { contextWindow: 256000, maxTokens: 32000, input: ["text"] },
  // Tencent Hunyuan
  "hy3": { contextWindow: 256000, maxTokens: 32000, input: ["text"] },
  // Xiaomi MiMo
  "mimo-v2.5": { contextWindow: 256000, maxTokens: 32000, input: ["text"] },
  "mimo-v2.5-pro": { contextWindow: 256000, maxTokens: 32000, input: ["text"] },
};

function familyCaps(id: string): Caps {
  if (id.startsWith("gpt-5")) return { contextWindow: 400000, maxTokens: 128000, input: ["text", "image"] };
  if (id.startsWith("claude-")) return { contextWindow: 200000, maxTokens: 64000, input: ["text", "image"] };
  if (id.startsWith("gemini-")) return { contextWindow: 1000000, maxTokens: 65536, input: ["text", "image"] };
  if (id.startsWith("deepseek-v4")) return { contextWindow: 128000, maxTokens: 32000, input: ["text"] };
  if (id.startsWith("glm-5")) return { contextWindow: 128000, maxTokens: 32000, input: ["text", "image"] };
  if (id.startsWith("kimi-")) return { contextWindow: 256000, maxTokens: 32000, input: ["text"] };
  if (id.startsWith("qwen3.8")) return { contextWindow: 256000, maxTokens: 32000, input: ["text", "image"] };
  if (id.startsWith("minimax-")) return { contextWindow: 256000, maxTokens: 32000, input: ["text"] };
  if (id === "hy3") return { contextWindow: 256000, maxTokens: 32000, input: ["text"] };
  if (id.startsWith("mimo-")) return { contextWindow: 256000, maxTokens: 32000, input: ["text"] };
  return { contextWindow: DEFAULT_CONTEXT, maxTokens: DEFAULT_MAX_TOKENS, input: ["text"] };
}

function capsFor(id: string): Caps {
  return MODEL_CAPS[id] ?? familyCaps(id);
}

function chatModel(id: string) {
  const chatOnly = CHAT_ONLY_IDS.has(id);
  const reasoning = !chatOnly && REASONING_IDS.has(id);
  const free = FREE_IDS.has(id);
  const caps = capsFor(id);
  return {
    id,
    name: id,
    reasoning,
    input: [...caps.input],
    cost: { ...ZERO_COST },
    contextWindow: caps.contextWindow,
    maxTokens: caps.maxTokens,
    // Private metadata (pi ignores unknown fields); consumed by streamBai and /bai-models.
    ...(chatOnly ? { baiChatOnly: true } : {}),
    ...(free ? { baiFree: true } : {}),
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
  return data.filter((m: any) => m && m.id).map((m: any) => {
    const id = String(m.id);
    return chatModel(id);
  });
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
function streamResponsesChat(model: any, context: any, options: any, apiKey: string) {
  const resolved = { ...model, baseUrl: API_BASE, api: "openai-responses" };
  return openAIResponsesApi().streamSimple(resolved, context, { ...options, apiKey });
}

// Resolve a model for the OpenAI Chat Completions core. B.AI's upstream rejects
// the `developer` message role (HTTP 400, code 1214 "角色信息不正确"), so we force
// the system prompt to use `role: "system"` by disabling developer-role support
// in the model's compat. (model.compat overrides the auto-detected compat.)
function resolveChatModel(model: any) {
  return {
    ...model,
    baseUrl: API_BASE,
    api: "openai-completions",
    compat: { ...(model.compat ?? {}), supportsDeveloperRole: false },
  };
}

// Events that prove the Responses stream is actually producing output (so we
// can commit to it instead of falling back to Chat Completions).
const PRODUCTIVE_EVENTS = new Set([
  "thinking_start", "thinking_delta",
  "text_start", "text_delta",
  "message_start", "message_delta", "content_delta",
  "assistant-message", "assistant-message-delta",
  "tool_call_start", "tool_call_delta", "tool_call_end",
  "done",
]);

function isEndpointUnsupported(ev: any): boolean {
  if (!ev || ev.type !== "error") return false;
  const parts = [
    ev.errorMessage, ev.message,
    ev.error?.message, ev.error?.errorMessage, ev.error?.code,
    ev.error ? JSON.stringify(ev.error) : "",
  ];
  const hay = parts.filter(Boolean).join(" ").toLowerCase();
  return hay.includes("model_not_supported_on_endpoint") || hay.includes("not supported on /v1/responses");
}

// Try the Responses API; if B.AI rejects the model as /v1/responses-only, retry
// transparently on /v1/chat/completions (which supports every model). Reasoning
// params are dropped on the fallback since chat/completions uses native
// reasoning tokens instead of the Responses reasoning object.
async function* chatWithResponsesFallback(model: any, context: any, options: any, apiKey: string) {
  const responsesStream = streamResponsesChat(model, context, options, apiKey);
  const it = (responsesStream as any)[Symbol.asyncIterator]();
  const buffer: any[] = [];
  let committed = false;
  while (true) {
    const { value: ev, done } = await it.next();
    if (done) break;
    if (ev?.type === "error") {
      if (isEndpointUnsupported(ev)) {
        const chatResolved = resolveChatModel(model);
        const chatOptions = { ...options, apiKey };
        delete chatOptions.reasoning;
        delete chatOptions.summary;
        delete chatOptions.thinkingLevel;
        yield* (openAICompletionsApi().streamSimple(chatResolved, context, chatOptions) as any);
        return;
      }
      for (const b of buffer) yield b;
      yield ev;
      return;
    }
    buffer.push(ev);
    if (PRODUCTIVE_EVENTS.has(ev.type)) {
      for (const b of buffer) yield b;
      committed = true;
      break;
    }
  }
  if (committed) {
    while (true) {
      const { value: ev, done } = await it.next();
      if (done) break;
      yield ev;
    }
  }
}

function streamBai(model: any, context: any, options: any) {
  // pi may drop unknown metadata fields (baiImageModel / baiChatOnly / baiFree)
  // when it registers models, so fall back to id-based lookups against our
  // curated sets.
  if (model.baiImageModel || IMAGE_SEED_IDS.includes(model.id)) return streamImageGeneration(model, context, options);
  const apiKey = process.env[API_KEY_ENV];
  // Models known to be chat/completions-only (e.g. the limited-time-free
  // deepseek-v4-flash / hy3 / mimo-v2.5 families, glm-5.3-flash) are routed
  // straight to the openai-completions core — no wasted /v1/responses call.
  if (model.baiChatOnly || CHAT_ONLY_IDS.has(model.id)) {
    return observe(model.id, openAICompletionsApi().streamSimple(resolveChatModel(model), context, { ...options, apiKey }));
  }
  // All other models go through the Responses API (native reasoning effort /
  // summary). B.AI's catalog metadata does NOT advertise which models support
  // /v1/responses, so we attempt it and transparently retry on
  // /v1/chat/completions when the model is rejected there. The returned stream
  // is observed so the model's access tier is learned from the outcome.
  if (lazyStream) return observe(model.id, lazyStream(model, async () => chatWithResponsesFallback(model, context, options, apiKey)));
  return streamResponsesChat(model, context, options, apiKey);
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
  return m.baiImageModel || IMAGE_SEED_IDS.includes(m.id) ? "image" : "chat";
}

function fmtSize(n?: number): string {
  if (!n) return "—";
  if (n >= 1_000_000) return `${Math.round(n / 1_000_000)}M`;
  if (n >= 1000) return `${Math.round(n / 1000)}K`;
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
  // Free-tier models first, then alphabetical — makes the usable set obvious.
  // Access tier is learned from live responses (see modelStatus) and falls back
  // to the seeded FREE_IDS set for models not yet observed.
  const tierLabel = (m: any) => {
    const t = accessTier(m.id);
    if (t === "free") return "**FREE**";
    if (t === "premium") return "CHARGE";
    if (t === "quota") return "QUOTA";
    return "—";
  };
  const isFree = (m: any) => accessTier(m.id) === "free";
  const sorted = [...models].sort(
    (a, b) => (isFree(b) ? 1 : 0) - (isFree(a) ? 1 : 0) || String(a.id).localeCompare(String(b.id)),
  );
  const rows = sorted
    .map((m) => {
      const isImage = modelType(m) === "image";
      const ctx = isImage ? "—" : fmtSize(m.contextWindow);
      const max = isImage ? "—" : fmtSize(m.maxTokens);
      const mods = isImage
        ? "image"
        : (m.input ?? ["text"]).map((x: string) => (x === "image" ? "image" : "text")).join("+");
      return `| \`${m.id}\` | ${modelType(m)} | ${tierLabel(m)} | ${m.reasoning ? "✓" : "—"} | ${ctx} | ${max} | ${mods} |`;
    })
    .join("\n");
  return [
    ...head,
    "",
    "| Model | Type | Access | Reasoning | Context | Max Out | Modalities |",
    "|---|---|:---:|:---:|:---:|:---:|:---:|",
    rows,
    "",
    "_Context windows / max output come from each model's public specs (B.AI's `/v1/models` does not expose them); the table falls back to the family default for any unlisted model. Modalities lists input capabilities for chat models and `image` for image models._",
    "_**FREE** = usable on the free tier (200). **CHARGE** = 403, deposit required. **QUOTA** = 400, insufficient quota. Access is learned from your live chat responses and refreshed on demand with `/bai-free`; unknown models fall back to the seeded free list._",
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

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

// Probe a single model's access tier via a tiny non-streaming chat request.
// 200 -> free, 403 -> premium (charge), 400 insufficient_quota -> quota.
// Retries on 429/5xx with backoff; other statuses are "unknown".
async function probeAccess(id: string, apiKey: string, signal?: AbortSignal): Promise<string> {
  const body = { model: id, messages: [{ role: "user", content: "hi" }], max_tokens: 8, stream: false };
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const res = await fetch(`${API_BASE}/chat/completions`, {
        method: "POST",
        headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal,
      });
      if (res.status === 200) return "free";
      if (res.status === 403) return "premium";
      if (res.status === 400) {
        const txt = await res.text();
        return txt.includes("insufficient_user_quota") ? "quota" : "unknown";
      }
      if (res.status === 429 || res.status >= 500) { await sleep(8000); continue; }
      return "unknown";
    } catch {
      if (signal?.aborted) return "unknown";
      await sleep(5000); continue;
    }
  }
  return "unknown";
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

  // On-demand full refresh of access tiers. B.AI changes pricing over time
  // (free <-> charge), so this lets the user re-probe the whole catalog.
  pi.registerCommand("bai-free", {
    description: "Probe every B.AI model and refresh the FREE/CHARGE/QUOTA status (pricing changes over time).",
    handler: async (_args: string, ctx: any) => {
      const apiKey = process.env[API_KEY_ENV];
      const all = baiModels(ctx).filter((m: any) => !IMAGE_SEED_IDS.includes(m.id));
      const note = (msg: string, level: "info" | "warning" = "info") =>
        ctx?.mode === "tui" ? pi.appendEntry("bai-free", { markdown: msg }) : notifyOrPrint(ctx, msg, level);
      if (!apiKey) { note("_Set `BAI_API_KEY` to probe model access._", "warning"); return; }
      if (all.length === 0) { note("_No B.AI models available — set `BAI_API_KEY` and restart pi._", "warning"); return; }
      note(`_Probing ${all.length} B.AI model(s) for access tier… (this may take a minute)_`);
      const results: Record<string, string[]> = { free: [], premium: [], quota: [], unknown: [] };
      for (const m of all) {
        if (ctx?.signal?.aborted) break;
        const tier = await probeAccess(m.id, apiKey, ctx?.signal);
        if (tier) modelStatus.set(m.id, tier);
        results[tier]?.push(m.id);
        await sleep(700);
      }
      if (ctx?.signal?.aborted) { note("_Probe aborted._", "warning"); return; }
      const fmt = (arr: string[]) => (arr.length ? arr.map((i) => `\`${i}\``).join(", ") : "—");
      const md = [
        "# B.AI model access",
        "",
        `_Probed ${all.length} model(s). Status is also learned automatically from your chat responses._`,
        "",
        `**FREE (${results.free.length})**: ${fmt(results.free)}`,
        `**CHARGE (${results.premium.length})**: ${fmt(results.premium)}`,
        `**QUOTA (${results.quota.length})**: ${fmt(results.quota)}`,
        results.unknown.length ? `**UNKNOWN (${results.unknown.length})**: ${fmt(results.unknown)}` : "",
      ].filter(Boolean).join("\n");
      note(md);
    },
  });
  pi.registerEntryRenderer?.("bai-free", (entry: any) =>
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
