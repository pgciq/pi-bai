# pi-bai

A standalone **pi** extension for [B.AI](https://b.ai/) — a unified LLM gateway whose
API is compatible with the **OpenAI Chat Completions**, **OpenAI Responses**, and
**Anthropic Messages** protocols.

- **Chat** is routed through the **OpenAI Responses API** (`POST /v1/responses`), which
  exposes native **reasoning** controls (`reasoning.effort` / `reasoning.summary`). The
  extension delegates to pi-ai's built-in `openai-responses` core, so streaming, tool
  calls, thinking blocks, and reasoning all work out of the box.
- **Image generation** uses the OpenAI-compatible `POST /v1/images/generations`
  (`gpt-image-2`, `grok-imagine-image-2.0`), saved under `.pi/generated-images/`.

> Docs: https://docs.b.ai/llmservice/api/

## Setup

```bash
# macOS / Linux / WSL
export BAI_API_KEY="sk-..."

# Windows PowerShell
$env:BAI_API_KEY = "sk-..."
```

Install like any pi extension:

```bash
npm install pi-bai
```

Then restart pi so the `bai` provider is loaded. The model catalog is fetched
automatically from `GET https://api.b.ai/v1/models` on startup and overrides the
built-in seed list.

## How it works

B.AI exposes three protocol-compatible endpoints under the production base URL
`https://api.b.ai/v1`:

| Endpoint | Protocol | Used here |
|---|---|---|
| `/v1/chat/completions` | OpenAI Chat Completions | ✅ chat for chat-only models |
| `/v1/responses` | OpenAI Responses | ✅ chat for reasoning-capable models |
| `/v1/messages` | Anthropic Messages | — |
| `/v1/images/generations` | OpenAI Images | ✅ image generation |

- **Auth** — `BAI_API_KEY` sent as `Authorization: Bearer <key>`.
- **Reasoning** — chat models flagged `reasoning: true` (GPT-5.x, Claude Opus 5,
  Gemini 3.6, DeepSeek v4, GLM-5.3, Kimi, Grok, MiMo, MiniMax, Hunyuan) get a
  `thinkingLevelMap` so pi's thinking-level selector maps to B.AI's
  `none / low / medium / high / xhigh / max` efforts. By default reasoning is
  `none`; selecting a level in pi sends `reasoning: { effort, summary }`.
- **Per-model endpoint selection** — B.AI rejects some models on `/v1/responses`
  with `model_not_supported_on_endpoint`. The chat/completions-only models we know
  about (`deepseek-v4-flash`, `deepseek-v4-flash-vision-exp`, `hy3`, `mimo-v2.5`,
  `glm-5.3-flash`) are flagged `baiChatOnly: true` and routed straight to the
  OpenAI Chat Completions core. Every other model is tried on the Responses API
  first (for native reasoning effort / summary), and **if B.AI rejects it as
  `/v1/responses`-only, the request is transparently retried on `/v1/chat/completions`**
  (which supports all models; reasoning params are dropped on the retry since those
  models reason via native tokens). So you don't need to enumerate chat-only models
  — the fallback covers any we haven't listed. Chat-only models still reason
  natively (the core surfaces their `reasoning_content` thinking tokens); they just
  don't expose B.AI's Responses-API effort/summary controls.
- **System role** — B.AI's upstream rejects the OpenAI `developer` message role
  (HTTP 400, code 1214, "角色信息不正确"). Every chat/completions request is forced
  to use `role: "system"` for the system prompt (via `compat.supportsDeveloperRole: false`
  on the resolved model), so reasoning-capable chat-only models that hit the
  Responses→Chat-Completions fallback also work.
- **Model catalog** — `GET /v1/models` returns an OpenAI-compatible
  `{ object, success, data: [{ id, object, created }] }`. `refreshModels` fetches it
  on startup / on demand, publishes the result, and always merges the image models
  (which are not part of the `/v1/models` catalog).
- **Free tier & access status** — B.AI does **not** advertise free/premium in
  `/v1/models`, and **pricing changes over time** (free ⇄ charge). The signal is the
  API response, and the extension **learns it automatically**:
  - `200` from `/v1/chat/completions` → **FREE**
  - `403 access_denied` ("Deposit required to unlock premium models") → **CHARGE**
  - `400 insufficient_user_quota` → **QUOTA**
  Every chat stream is observed, so the moment you use a model that flipped
  free→charge (or charge→free), the `/bai-models` badge corrects itself. Unknown
  models fall back to the seeded free list below.
  The currently-free models (confirmed by probing the catalog) are:
  `deepseek-v4-flash`, `deepseek-v4-flash-vision-exp`, `hy3`, `mimo-v2.5`,
  `glm-5.3-flash`, `qwen3.8-flash`. For a full refresh on demand, run `/bai-free`
  (it probes every model against `/v1/chat/completions` and updates all badges).
- **Image generation** — dispatched from `streamBai` via the `baiImageModel` flag,
  the same pattern `pi-cloudflare-workers-ai` uses (pi's extension API has no separate
  image-registration surface). The exact image endpoint is assumed OpenAI-compatible;
  change `IMAGE_GEN_PATH` in `extensions/bai.ts` if your plan differs.

## Models

The seed list covers B.AI's language-model families and image models. After the first
refresh, the live `GET /v1/models` catalog replaces the chat seed (image models are
always merged back in), so you always see exactly the models enabled for your API key.

**Capabilities.** B.AI's `GET /v1/models` returns only `{ id, object, created }` — it does
**not** expose context windows, max output, or modalities. The extension therefore ships a
per-model capability table (`MODEL_CAPS` in `extensions/bai.ts`) sourced from each model's
public specifications, with a family-based fallback (`familyCaps`) for any model the live
catalog adds later. The `/bai-models` command shows **Context**, **Max Out**, and
**Modalities** columns so you can see each model's real limits at a glance:

| Family | Context | Max out | Input |
|---|---|---|---|
| `gpt-5.*` | 400K | 16K–128K | text + image |
| `claude-*` | 200K | 64K | text + image |
| `gemini-*` | 1M | 64K | text + image |
| `deepseek-v4*` | 128K | 32K | text (+ image for vision-exp/pro) |
| `glm-5.*` | 128K | 32K | text + image |
| `kimi-*`, `qwen3.8*`, `minimax-*`, `hy3`, `mimo-*` | 256K | 32K | text |

> These are best-effort figures from public model specs. If B.AI's actual limits differ,
> edit `MODEL_CAPS` (or report it) — they are not fetched from the API.

## Commands

## Commands

### `/bai-models`

Lists the B.AI models currently available to pi. By default it shows only the
confirmed **FREE** chat models and hides charged / quota models (image models are
listed under `all`, since their pricing isn't exposed by B.AI):

```
/bai-models
```

Pass `all` to list every model, including charged, quota-limited, and image models:

```
/bai-models all
```

### `/bai-docs`

Opens the B.AI API reference in your browser.

```
/bai-docs
```

### `/bai-free`

Probes **every** B.AI model against `/v1/chat/completions` and refreshes the
FREE / CHARGE / QUOTA status for all of them (pricing changes over time, so this
re-syncs the badges). It reports the counts and the model ids in each tier.

```
/bai-free
```

> The status is also learned automatically from your normal chat usage, so you
> only need `/bai-free` when you want an immediate full refresh.

## Notes

- Generated images are saved under `.pi/generated-images/` and reported as a clickable
  link (OSC 8 hyperlink); the TUI also renders an inline preview.
- If `BAI_API_KEY` is unset, pi marks the `bai` provider as unconfigured and the seed
  list is shown until a key is provided and a refresh succeeds.
- The image-generation endpoint shape was inferred from B.AI's "GPT-Image-2 is an
  OpenAI image generation … model" description (docs image pages were rate-limited at
  implementation time). Verify `IMAGE_GEN_PATH` against the live docs if needed.
