import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it, vi } from "vitest";
import {
  buildOptions,
  callJev,
  compactOptionText,
  JEV_INSTRUCTIONS,
  JEV_MODEL,
  JevCache,
  JevHttpError,
  type JevItem,
  JevRanker,
  type JevResponse,
  jevConfigFromArgs,
  optionText,
  parseAnswer,
  rankFromProbabilities,
  rerankWithJev,
  SpendCapError,
} from "./jev.js";

function item(id: string, extra: Partial<JevItem> = {}): JevItem {
  return { id, name: id, description: `does ${id}`, ...extra };
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status });
}

function choiceResponse(probabilities: Record<string, number>, inputTokens = 100): JevResponse {
  return {
    model: JEV_MODEL,
    answers: { select: { type: "choice", probabilities } },
    usage: { input_tokens: inputTokens, output_tokens: 1 },
  };
}

const noSleep = async () => {};

describe("optionText", () => {
  it("carries description plus the schema fields Ratel indexes (names, descriptions, enums, nested)", () => {
    const text = optionText(
      item("read_file", {
        description: "Read a file from disk",
        input_schema: {
          type: "object",
          properties: {
            path: { type: "string", description: "absolute path" },
            encoding: { type: "string", enum: ["utf8", "binary"], description: "file encoding" },
            opts: {
              type: "array",
              items: { type: "object", properties: { mode: { description: "open mode" } } },
            },
          },
        },
        output_schema: { properties: { content: { description: "file body" } } },
      }),
    );
    expect(text).toBe(
      "Read a file from disk\n" +
        "Parameters: path — absolute path; encoding — file encoding (one of: utf8, binary); " +
        "opts; opts[].mode — open mode\n" +
        "Returns: content — file body",
    );
  });

  it("is just the description for skills / schema-less items", () => {
    expect(optionText(item("s1", { description: "Build slides" }))).toBe("Build slides");
    expect(optionText(item("t1", { description: "x", input_schema: {} }))).toBe("x");
  });
});

describe("buildOptions", () => {
  const pool = Array.from({ length: 30 }, (_, i) => item(`t${i}`));

  it("is deterministic for the same seedKey and varies across seedKeys", () => {
    const a = buildOptions(pool, "sc1:jev:30", 42).order;
    expect(buildOptions(pool, "sc1:jev:30", 42).order).toEqual(a);
    expect(buildOptions(pool, "sc2:jev:30", 42).order).not.toEqual(a);
  });

  it("does not leave gold (pool position 0) first across scenarios", () => {
    const firsts = Array.from(
      { length: 50 },
      (_, i) => buildOptions(pool, `sc${i}:jev:30`, 42).order[0],
    );
    expect(firsts.filter((k) => k === "t0").length).toBeLessThan(10);
  });

  it("keys by name, suffixes duplicate names, and maps every key back to its id", () => {
    const dup = [item("a", { name: "search" }), item("b", { name: "search" }), item("c")];
    const { criteria, keyToId, order } = buildOptions(dup, "k", 1);
    expect(Object.keys(criteria).sort()).toEqual(["c", "search", "search#2"]);
    expect(order).toEqual(Object.keys(criteria));
    expect(new Set(order.map((k) => keyToId.get(k)))).toEqual(new Set(["a", "b", "c"]));
  });
});

describe("rankFromProbabilities", () => {
  const opts = {
    order: ["x", "y", "z", "w"],
    keyToId: new Map([
      ["x", "id-x"],
      ["y", "id-y"],
      ["z", "id-z"],
      ["w", "id-w"],
    ]),
  };

  it("ranks by probability desc and breaks ties in sent order", () => {
    const hits = rankFromProbabilities({ x: 0, y: 0.7, z: 0.3, w: 0 }, opts);
    expect(hits.map((h) => h.id)).toEqual(["id-y", "id-z", "id-x", "id-w"]);
    expect(hits[0].score).toBe(0.7);
  });

  it("scores omitted options 0 and ignores unknown keys", () => {
    const hits = rankFromProbabilities({ w: 1, bogus: 5 }, opts);
    expect(hits.map((h) => h.id)).toEqual(["id-w", "id-x", "id-y", "id-z"]);
    expect(hits).toHaveLength(4);
  });
});

describe("callJev", () => {
  const body = {
    model: JEV_MODEL,
    state: "q",
    questions: {
      select: { type: "choice" as const, instructions: "i", criteria: { a: "x" } },
    },
  };

  it("retries 429/529 then succeeds, sending the bearer key", async () => {
    const fetchImpl = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ error: "slow down" }, 429))
      .mockResolvedValueOnce(jsonResponse({ error: "overloaded" }, 529))
      .mockResolvedValueOnce(jsonResponse(choiceResponse({ a: 1 })));
    const { response } = await callJev(body, { apiKey: "k", fetchImpl, sleep: noSleep });
    expect(response.answers.select.probabilities).toEqual({ a: 1 });
    expect(fetchImpl).toHaveBeenCalledTimes(3);
    const init = fetchImpl.mock.calls[0][1] as RequestInit;
    expect((init.headers as Record<string, string>).Authorization).toBe("Bearer k");
    expect(JSON.parse(init.body as string)).toEqual(body);
  });

  it("fails fast on 401/422", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(jsonResponse({ error: "bad key" }, 401));
    await expect(callJev(body, { apiKey: "k", fetchImpl, sleep: noSleep })).rejects.toBeInstanceOf(
      JevHttpError,
    );
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });

  it("gives up after maxAttempts", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(jsonResponse({}, 529));
    await expect(
      callJev(body, { apiKey: "k", fetchImpl, sleep: noSleep, maxAttempts: 3 }),
    ).rejects.toThrow(/after 3 attempts/);
    expect(fetchImpl).toHaveBeenCalledTimes(3);
  });
});

describe("JevRanker", () => {
  const pool = [item("gold"), item("d1"), item("d2")];
  const ctx = { scenarioId: "sc", poolSize: 3, seedKey: "sc:jev:3" };

  it("calls the API once, caches to disk, and serves reruns from the cache", async () => {
    const cachePath = join(mkdtempSync(join(tmpdir(), "jev-")), "cache.jsonl");
    const fetchImpl = vi
      .fn()
      .mockResolvedValue(jsonResponse(choiceResponse({ gold: 0.9, d1: 0.1, d2: 0 }, 250)));
    const ranker = new JevRanker({
      kind: "tool",
      cache: new JevCache(cachePath),
      seed: 42,
      apiKey: () => "k",
      client: { fetchImpl, sleep: noSleep },
    });
    const first = await ranker.rank("find gold", pool, ctx);
    expect(first.hits.map((h) => h.id)).toEqual(["gold", "d1", "d2"]);
    expect(first.meta).toMatchObject({ jev_model: JEV_MODEL, input_tokens: 250, cache_hit: false });
    const sent = JSON.parse((fetchImpl.mock.calls[0][1] as RequestInit).body as string);
    expect(sent.questions.select.instructions).toBe(JEV_INSTRUCTIONS.tool);
    expect(sent.state).toBe("find gold");

    // A fresh ranker over the same file (a rerun) makes no network call.
    const noFetch = vi.fn();
    const rerun = new JevRanker({
      kind: "tool",
      cache: new JevCache(cachePath),
      seed: 42,
      apiKey: () => undefined,
      client: { fetchImpl: noFetch },
    });
    const second = await rerun.rank("find gold", pool, ctx);
    expect(noFetch).not.toHaveBeenCalled();
    expect(second.hits).toEqual(first.hits);
    expect(second.meta.cache_hit).toBe(true);
    expect(rerun.stats).toMatchObject({ calls: 0, cacheHits: 1, inputTokens: 0 });
    expect(readFileSync(cachePath, "utf-8").trim().split("\n")).toHaveLength(1);
  });

  it("requires TYPESAFE_API_KEY only for uncached requests", async () => {
    const cachePath = join(mkdtempSync(join(tmpdir(), "jev-")), "cache.jsonl");
    const ranker = new JevRanker({
      kind: "skill",
      cache: new JevCache(cachePath),
      seed: 42,
      apiKey: () => undefined,
    });
    const saved = process.env.TYPESAFE_API_KEY;
    delete process.env.TYPESAFE_API_KEY;
    try {
      await expect(ranker.rank("q", pool, ctx)).rejects.toThrow(/TYPESAFE_API_KEY/);
    } finally {
      if (saved !== undefined) process.env.TYPESAFE_API_KEY = saved;
    }
  });

  it("rejects a response without probabilities and does not cache it", async () => {
    const cachePath = join(mkdtempSync(join(tmpdir(), "jev-")), "cache.jsonl");
    const fetchImpl = vi
      .fn()
      .mockResolvedValue(jsonResponse({ model: JEV_MODEL, answers: {}, usage: {} }));
    const ranker = new JevRanker({
      kind: "tool",
      cache: new JevCache(cachePath),
      seed: 42,
      apiKey: () => "k",
      client: { fetchImpl, sleep: noSleep },
    });
    await expect(ranker.rank("q", pool, ctx)).rejects.toThrow(/no probabilities/);
    expect(new JevCache(cachePath).size).toBe(0);
  });
});

describe("rerankWithJev", () => {
  const pool = ["a", "b", "c", "d", "e"].map((id) => item(id));
  const ratelHits = [
    { id: "a", score: 9 },
    { id: "b", score: 5 },
    { id: "c", score: 1 },
    { id: "d", score: 0.5 },
  ];
  const ctx = { scenarioId: "sc", poolSize: 5, seedKey: "sc:ratel+jev:5" };

  function ranker(fetchImpl: typeof fetch) {
    const cachePath = join(mkdtempSync(join(tmpdir(), "jev-")), "cache.jsonl");
    return new JevRanker({
      kind: "tool",
      cache: new JevCache(cachePath),
      seed: 42,
      apiKey: () => "k",
      client: { fetchImpl, sleep: noSleep },
    });
  }

  it("sends only Ratel's top-depth shortlist and returns Jev's order over it", async () => {
    const fetchImpl = vi
      .fn()
      .mockResolvedValue(jsonResponse(choiceResponse({ a: 0.1, b: 0.2, c: 0.7 })));
    const { hits, meta } = await rerankWithJev(ranker(fetchImpl), "q", pool, ratelHits, 3, ctx);
    const sent = JSON.parse((fetchImpl.mock.calls[0][1] as RequestInit).body as string);
    expect(Object.keys(sent.questions.select.criteria).sort()).toEqual(["a", "b", "c"]);
    expect(hits.map((h) => h.id)).toEqual(["c", "b", "a"]);
    expect(meta).toMatchObject({ rerank_depth: 3, ratel_candidates: 3, cache_hit: false });
  });

  it("passes a shortlist of < 2 through without calling Jev", async () => {
    const fetchImpl = vi.fn();
    const one = await rerankWithJev(ranker(fetchImpl), "q", pool, [ratelHits[0]], 20, ctx);
    const none = await rerankWithJev(ranker(fetchImpl), "q", pool, [], 20, ctx);
    expect(fetchImpl).not.toHaveBeenCalled();
    expect(one.hits).toEqual([ratelHits[0]]);
    expect(one.meta).toEqual({ rerank_depth: 20, ratel_candidates: 1 });
    expect(none.hits).toEqual([]);
  });
});

describe("jevConfigFromArgs + self-hosted servers", () => {
  const args = (m: Record<string, string>) => (name: string, fallback: string) =>
    m[name] ?? fallback;

  it("defaults reproduce TypeSafe Jev (pinned model, key required, no extras)", () => {
    expect(jevConfigFromArgs(args({}))).toEqual({
      model: JEV_MODEL,
      extraBody: undefined,
      requireApiKey: true,
      client: undefined,
    });
  });

  it("--jev-base-url targets /v1/systemone keylessly; --jev-body-extra must be an object", () => {
    const cfg = jevConfigFromArgs(
      args({
        "--jev-base-url": "http://localhost:8000/",
        "--jev-model": "english",
        "--jev-body-extra": '{"head_max_len":1024}',
      }),
    );
    expect(cfg).toEqual({
      model: "english",
      extraBody: { head_max_len: 1024 },
      requireApiKey: false,
      client: { endpoint: "http://localhost:8000/v1/systemone" },
    });
    expect(() => jevConfigFromArgs(args({ "--jev-body-extra": "[1]" }))).toThrow(/JSON object/);
  });

  it("sends no Authorization header and includes extras in the body", async () => {
    const cachePath = join(mkdtempSync(join(tmpdir(), "jev-")), "cache.jsonl");
    const fetchImpl = vi.fn().mockResolvedValue(jsonResponse(choiceResponse({ a: 0.6, b: 0.4 })));
    const ranker = new JevRanker({
      kind: "tool",
      cache: new JevCache(cachePath),
      seed: 42,
      model: "english",
      extraBody: { head_max_len: 1024 },
      requireApiKey: false,
      apiKey: () => undefined,
      client: { endpoint: "http://localhost:8000/v1/systemone", fetchImpl, sleep: noSleep },
    });
    const saved = process.env.TYPESAFE_API_KEY;
    delete process.env.TYPESAFE_API_KEY;
    try {
      await ranker.rank("q", [item("a"), item("b")], {
        scenarioId: "s",
        poolSize: 2,
        seedKey: "k",
      });
    } finally {
      if (saved !== undefined) process.env.TYPESAFE_API_KEY = saved;
    }
    const [url, init] = fetchImpl.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("http://localhost:8000/v1/systemone");
    expect(init.headers).not.toHaveProperty("Authorization");
    const sent = JSON.parse(init.body as string);
    expect(sent).toMatchObject({ model: "english", head_max_len: 1024 });
  });
});

describe("tournament for option-capped servers (OpenJev/Verdict)", () => {
  // Server mock: probability ∝ the number in the key, so `t{max}` is always the best option.
  function scoringFetch() {
    const sizes: number[] = [];
    const fetchImpl = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
      const body = JSON.parse(init?.body as string);
      const keys = Object.keys(body.questions.select.criteria);
      sizes.push(keys.length);
      const w = keys.map((k) => Number(k.slice(1)) + 1);
      const total = w.reduce((a, b) => a + b, 0);
      return jsonResponse(
        choiceResponse(Object.fromEntries(keys.map((k, i) => [k, w[i] / total])), keys.length),
      );
    });
    return { fetchImpl, sizes };
  }
  function ranker(fetchImpl: typeof fetch, maxOptions?: number) {
    const cachePath = join(mkdtempSync(join(tmpdir(), "jev-")), "cache.jsonl");
    return new JevRanker({
      kind: "tool",
      cache: new JevCache(cachePath),
      seed: 42,
      requireApiKey: false,
      apiKey: () => undefined,
      maxOptions,
      client: { fetchImpl, sleep: noSleep },
    });
  }
  const pool = (n: number) => Array.from({ length: n }, (_, i) => item(`t${i}`));
  const ctx = (n: number) => ({ scenarioId: "sc", poolSize: n, seedKey: `sc:jev:${n}` });

  it("pool 30, cap 15: 2 chunks + final = 3 calls, all within the cap, best first", async () => {
    const { fetchImpl, sizes } = scoringFetch();
    const { hits, meta } = await ranker(fetchImpl, 15).rank("q", pool(30), ctx(30));
    expect(sizes).toEqual([15, 15, 10]);
    expect(hits.map((h) => h.id).slice(0, 5)).toEqual(["t29", "t28", "t27", "t26", "t25"]);
    expect(new Set(hits.map((h) => h.id)).size).toBe(30);
    expect(meta).toMatchObject({ tournament: true, calls: 3, cache_hit: false, input_tokens: 40 });
  });

  it("pool 100, cap 15: 7 + 3 + 1 = 11 calls, every request ≤ 15 options", async () => {
    const { fetchImpl, sizes } = scoringFetch();
    const { hits, meta } = await ranker(fetchImpl, 15).rank("q", pool(100), ctx(100));
    expect(sizes).toHaveLength(11);
    expect(Math.max(...sizes)).toBeLessThanOrEqual(15);
    expect(hits[0].id).toBe("t99");
    expect(hits).toHaveLength(100);
    expect(meta.calls).toBe(11);
  });

  it("is deterministic and a rerun is served from cache", async () => {
    const { fetchImpl } = scoringFetch();
    const r = ranker(fetchImpl, 15);
    const a = await r.rank("q", pool(30), ctx(30));
    const b = await r.rank("q", pool(30), ctx(30));
    expect(b.hits).toEqual(a.hits);
    expect(b.meta.cache_hit).toBe(true);
    expect(fetchImpl).toHaveBeenCalledTimes(3);
  });

  it("no cap, or a pool within the cap, is one plain call with no tournament fields", async () => {
    const { fetchImpl, sizes } = scoringFetch();
    const one = await ranker(fetchImpl, 24).rank("q", pool(20), ctx(20));
    const none = await ranker(fetchImpl).rank("q", pool(30), ctx(30));
    expect(sizes).toEqual([20, 30]);
    expect(one.meta).not.toHaveProperty("tournament");
    expect(none.meta).not.toHaveProperty("calls");
  });

  it("terminates even when chunks are barely above the keep size", async () => {
    const { fetchImpl, sizes } = scoringFetch();
    const { hits } = await ranker(fetchImpl, 6).rank("q", pool(7), ctx(7));
    expect(hits[0].id).toBe("t6");
    expect(Math.max(...sizes)).toBeLessThanOrEqual(6);
  });
});

describe("compact option text and flags", () => {
  it("is name + first sentence, no parameters", () => {
    expect(
      compactOptionText(
        item("calc", {
          name: "geometry.area_triangle",
          description: "Calculate the area of a triangle. Uses base and height.\nMore.",
          input_schema: { properties: { base: { description: "the base" } } },
        }),
      ),
    ).toBe("geometry.area_triangle: Calculate the area of a triangle.");
  });

  it("--jev-option-text / --jev-max-options parse and validate", () => {
    const a = (m: Record<string, string>) => (n: string, f: string) => m[n] ?? f;
    expect(
      jevConfigFromArgs(a({ "--jev-option-text": "compact", "--jev-max-options": "15" })),
    ).toMatchObject({ optionText: "compact", maxOptions: 15 });
    expect(() => jevConfigFromArgs(a({ "--jev-option-text": "short" }))).toThrow(
      /standard or compact/,
    );
    expect(() => jevConfigFromArgs(a({ "--jev-max-options": "5" }))).toThrow(/integer > 5/);
  });
});

describe("OpenAI Decisions protocol", () => {
  const pool = [item("a"), item("b"), item("c")];
  const ctx = { scenarioId: "sc", poolSize: 3, seedKey: "sc:jev:3" };

  function openaiResponse(probs: Record<string, number> | "refusal") {
    return {
      model: "gpt-6-luna",
      answers: [
        probs === "refusal"
          ? { type: "refusal", name: "select" }
          : {
              type: "choice",
              name: "select",
              choice: Object.entries(probs).sort((x, y) => y[1] - x[1])[0][0],
              probabilities: Object.entries(probs).map(([value, probability]) => ({
                value,
                probability,
              })),
              confidence: 0.5,
            },
      ],
      usage: { input_tokens: 42 },
    };
  }

  function ranker(fetchImpl: typeof fetch, cachePath?: string) {
    return new JevRanker({
      kind: "tool",
      cache: new JevCache(cachePath ?? join(mkdtempSync(join(tmpdir(), "jev-")), "cache.jsonl")),
      seed: 42,
      protocol: "openai",
      apiKey: () => "sk-test",
      client: { fetchImpl, sleep: noSleep },
    });
  }

  it("sends the Decisions request shape: input + questions[] with choices in sent order", async () => {
    const fetchImpl = vi
      .fn()
      .mockResolvedValue(jsonResponse(openaiResponse({ a: 0.2, b: 0.7, c: 0.1 })));
    const { hits, meta } = await ranker(fetchImpl).rank("find b", pool, ctx);
    const [url, init] = fetchImpl.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("https://api.openai.com/v1/decisions");
    expect((init.headers as Record<string, string>).Authorization).toBe("Bearer sk-test");
    const body = JSON.parse(init.body as string);
    expect(body.model).toBe("gpt-6-luna");
    expect(body.input).toBe("find b");
    expect(body).not.toHaveProperty("state");
    const q = body.questions[0];
    expect(q).toMatchObject({
      type: "choice",
      name: "select",
      instructions: JEV_INSTRUCTIONS.tool,
    });
    // same shuffled order and option text as the Jev request would use
    const jevOrder = buildOptions(pool, ctx.seedKey, 42);
    expect(q.choices.map((c: { value: string }) => c.value)).toEqual(jevOrder.order);
    expect(q.choices[0].description).toBe(jevOrder.criteria[jevOrder.order[0]]);
    expect(hits.map((h) => h.id)).toEqual(["b", "a", "c"]);
    expect(meta).toMatchObject({ jev_model: "gpt-6-luna", input_tokens: 42, cache_hit: false });
    expect(meta).not.toHaveProperty("refusal");
  });

  it("a refusal is cached, counted, and ranked as all-zero (sent order)", async () => {
    const cachePath = join(mkdtempSync(join(tmpdir(), "jev-")), "cache.jsonl");
    const fetchImpl = vi.fn().mockResolvedValue(jsonResponse(openaiResponse("refusal")));
    const r = ranker(fetchImpl, cachePath);
    const first = await r.rank("q", pool, ctx);
    expect(first.meta.refusal).toBe(true);
    expect(first.hits.map((h) => h.id)).toEqual(
      buildOptions(pool, ctx.seedKey, 42).order.map((k) => k),
    );
    expect(r.stats.refusals).toBe(1);
    const again = await ranker(vi.fn(), cachePath).rank("q", pool, ctx);
    expect(again.meta).toMatchObject({ refusal: true, cache_hit: true });
  });

  it("rejects a malformed response without caching it", async () => {
    const cachePath = join(mkdtempSync(join(tmpdir(), "jev-")), "cache.jsonl");
    const fetchImpl = vi.fn().mockResolvedValue(jsonResponse({ answers: [] }));
    await expect(ranker(fetchImpl, cachePath).rank("q", pool, ctx)).rejects.toThrow(
      /no probabilities/,
    );
    expect(new JevCache(cachePath).size).toBe(0);
  });

  it("--decision-api openai sets protocol and the gpt-6-luna default; jev default unchanged", () => {
    const a = (m: Record<string, string>) => (n: string, f: string) => m[n] ?? f;
    expect(jevConfigFromArgs(a({ "--decision-api": "openai" }))).toMatchObject({
      protocol: "openai",
      model: "gpt-6-luna",
      requireApiKey: true,
    });
    expect(jevConfigFromArgs(a({}))).not.toHaveProperty("protocol");
    expect(() => jevConfigFromArgs(a({ "--decision-api": "gemini" }))).toThrow(/jev or openai/);
  });

  it("parseAnswer maps the probabilities array to a key → probability record", () => {
    expect(
      parseAnswer("openai", openaiResponse({ x: 0.25, y: 0.75 }) as never, {
        scenarioId: "s",
        poolSize: 2,
      }),
    ).toEqual({ probabilities: { x: 0.25, y: 0.75 }, refusal: false });
  });
});

describe("provider key env, cost accounting and the dollar cap", () => {
  const pool = [item("a"), item("b")];
  const ctx = (i: number) => ({ scenarioId: `s${i}`, poolSize: 2, seedKey: `s${i}:jev:2` });
  const withUsage = (inTok: number, outTok: number) =>
    jsonResponse({
      ...choiceResponse({ a: 0.7, b: 0.3 }, inTok),
      usage: { input_tokens: inTok, output_tokens: outTok },
    });

  it("--jev-api-key-env names the key variable and makes a base-url server keyed", async () => {
    const a = (m: Record<string, string>) => (n: string, f: string) => m[n] ?? f;
    const cfg = jevConfigFromArgs(
      a({
        "--jev-base-url": "https://sage.levanto.ai",
        "--jev-model": "levanto-sage-v1.3",
        "--jev-api-key-env": "SAGE_API_KEY",
        "--jev-price-in": "0.05",
        "--jev-price-out": "10",
        "--jev-max-spend": "1.5",
      }),
    );
    expect(cfg).toMatchObject({
      model: "levanto-sage-v1.3",
      requireApiKey: true,
      apiKeyEnv: "SAGE_API_KEY",
      priceIn: 0.05 / 1e6,
      priceOut: 10 / 1e6,
      maxSpend: 1.5,
      client: { endpoint: "https://sage.levanto.ai/v1/systemone" },
    });
    expect(() => jevConfigFromArgs(a({ "--jev-max-spend": "-1" }))).toThrow(/≥ 0/);

    const fetchImpl = vi.fn().mockResolvedValue(withUsage(100, 1));
    const saved = process.env.SAGE_API_KEY;
    process.env.SAGE_API_KEY = "lv_live_test";
    try {
      const r = new JevRanker({
        kind: "tool",
        cache: new JevCache(join(mkdtempSync(join(tmpdir(), "jev-")), "c.jsonl")),
        seed: 42,
        ...cfg,
        client: { ...cfg.client, fetchImpl, sleep: noSleep },
      });
      await r.rank("q", pool, ctx(0));
      const init = fetchImpl.mock.calls[0][1] as RequestInit;
      expect((init.headers as Record<string, string>).Authorization).toBe("Bearer lv_live_test");
      expect(JSON.parse(init.body as string)).toMatchObject({
        model: "levanto-sage-v1.3",
        state: "q",
      });
      expect(r.stats).toMatchObject({ inputTokens: 100, outputTokens: 1 });
      expect(r.spent()).toBeCloseTo(100 * 0.05e-6 + 1 * 10e-6, 12);
    } finally {
      if (saved === undefined) delete process.env.SAGE_API_KEY;
      else process.env.SAGE_API_KEY = saved;
    }
  });

  it("a named key env that is unset fails fast", async () => {
    delete process.env.SAGE_TEST_MISSING;
    const r = new JevRanker({
      kind: "tool",
      cache: new JevCache(join(mkdtempSync(join(tmpdir(), "jev-")), "c.jsonl")),
      seed: 42,
      apiKeyEnv: "SAGE_TEST_MISSING",
      client: { endpoint: "https://example.invalid/v1/systemone", fetchImpl: vi.fn() },
    });
    await expect(r.rank("q", pool, ctx(0))).rejects.toThrow(/SAGE_TEST_MISSING is not set/);
  });

  it("the cap stops before the next uncached call but never blocks cache hits", async () => {
    const cachePath = join(mkdtempSync(join(tmpdir(), "jev-")), "c.jsonl");
    const fetchImpl = vi.fn(async () => withUsage(1_000_000, 0)); // $0.05/call, fresh body each time
    const mk = () =>
      new JevRanker({
        kind: "tool",
        cache: new JevCache(cachePath),
        seed: 42,
        apiKey: () => "k",
        priceIn: 0.05 / 1e6,
        maxSpend: 0.08,
        client: { fetchImpl, sleep: noSleep },
      });
    const r = mk();
    await r.rank("q0", pool, ctx(0)); // $0.05
    await r.rank("q1", pool, ctx(1)); // spent $0.05 < cap → allowed; now $0.10 ≥ cap
    await expect(r.rank("q2", pool, ctx(2))).rejects.toBeInstanceOf(SpendCapError);
    expect(fetchImpl).toHaveBeenCalledTimes(2);
    // a fresh run (spend resets to 0 for new calls) replays both cached answers for free
    const again = mk();
    await again.rank("q0", pool, ctx(0));
    await again.rank("q1", pool, ctx(1));
    expect(again.stats).toMatchObject({ calls: 0, cacheHits: 2 });
    expect(r.summary()).toMatch(/of \$0\.08 cap/);
  });
});
