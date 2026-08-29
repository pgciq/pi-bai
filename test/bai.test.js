import assert from "node:assert/strict";
import test from "node:test";
import extension from "../extensions/bai.ts";

function getProviderConfig() {
  let config;
  const commands = [];
  extension({
    registerProvider(_name, providerConfig) {
      config = providerConfig;
    },
    registerEntryRenderer() {},
    appendEntry() {},
    registerCommand(name, def) {
      commands.push({ name, def });
    },
  });
  return { config, commands };
}

test("registers the bai provider with the production base and env key", () => {
  const { config } = getProviderConfig();
  assert.equal(config.name, "B.AI");
  assert.equal(config.baseUrl, "https://api.b.ai/v1");
  assert.equal(config.apiKey, "$BAI_API_KEY");
  assert.equal(config.api, "openai-responses");
  assert.equal(typeof config.streamSimple, "function");
  assert.equal(typeof config.refreshModels, "function");
});

test("seeds chat + image models and flags reasoning/image metadata", () => {
  const { config } = getProviderConfig();
  assert.ok(Array.isArray(config.models) && config.models.length > 0);

  const ids = config.models.map((m) => m.id);
  for (const expected of ["gpt-5.6-sol", "claude-opus-5", "gemini-3.6-flash", "deepseek-v4-pro"]) {
    assert.ok(ids.includes(expected), `seed should include chat model ${expected}`);
  }

  const imageModels = config.models.filter((m) => m.baiImageModel);
  assert.ok(imageModels.length >= 1, "at least one image model is seeded");
  assert.ok(imageModels.every((m) => m.input.includes("text") && m.reasoning === false));

  const reasoningModel = config.models.find((m) => m.id === "gpt-5.6-sol");
  assert.ok(reasoningModel.reasoning, "gpt-5.6-sol should be flagged reasoning");
  assert.ok(reasoningModel.thinkingLevelMap && reasoningModel.thinkingLevelMap.off === "none");

  // Lighter tiers stay non-reasoning to avoid sending `reasoning` to unsupported models.
  const mini = config.models.find((m) => m.id === "gpt-5-mini");
  assert.equal(mini.reasoning, false);
});

test("registers /bai-models and /bai-docs commands", () => {
  const { commands } = getProviderConfig();
  const names = commands.map((c) => c.name);
  assert.ok(names.includes("bai-models"));
  assert.ok(names.includes("bai-docs"));
});

test("refreshes from the live /v1/models catalog and merges image models", async () => {
  const { config } = getProviderConfig();
  const fakeModels = [
    { id: "gpt-5.6-sol", object: "model", created: 1 },
    { id: "claude-opus-5", object: "model", created: 2 },
  ];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => ({
    ok: true,
    json: async () => ({ object: "list", success: true, data: fakeModels }),
  });

  let published = null;
  try {
    const result = await config.refreshModels({
      signal: new AbortController().signal,
      stored: undefined,
      publish: async (p) => {
        published = p;
      },
      allowNetwork: true,
    });
    // 2 fetched chat models + the merged image seed models.
    assert.ok(result.length >= 3);
    assert.ok(result.some((m) => m.id === "gpt-5.6-sol"));
    assert.ok(result.some((m) => m.baiImageModel), "image models are merged in");
    assert.ok(published && published.persist.provider === "bai");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("falls back to the cache (with image models) when the API is unreachable", async () => {
  const { config } = getProviderConfig();
  const cachedModels = [
    {
      id: "cached-model",
      name: "Cached model",
      reasoning: false,
      input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 200000,
      maxTokens: 64000,
      baiImageModel: true,
    },
  ];

  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => {
    throw new Error("offline");
  };

  try {
    const result = await config.refreshModels({
      signal: new AbortController().signal,
      stored: { provider: "bai", models: cachedModels },
      publish: async () => true,
    });
    assert.deepEqual(result, cachedModels);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("falls back to the seed list (chat + image) when offline with no cache", async () => {
  const { config } = getProviderConfig();
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => {
    throw new Error("offline");
  };

  try {
    const result = await config.refreshModels({
      signal: new AbortController().signal,
      stored: undefined,
      publish: async () => true,
    });
    assert.ok(Array.isArray(result));
    assert.ok(result.length > 0);
    assert.ok(result.some((m) => m.baiImageModel), "seed includes image models");
  } finally {
    globalThis.fetch = originalFetch;
  }
});
