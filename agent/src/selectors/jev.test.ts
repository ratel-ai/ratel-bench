import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it, vi } from "vitest";
import {
  buildOptions,
  callJev,
  JEV_INSTRUCTIONS,
  JEV_MODEL,
  JevCache,
  JevHttpError,
  type JevItem,
  JevRanker,
  type JevResponse,
  jevConfigFromArgs,
  optionText,
  rankFromProbabilities,
  rerankWithJev,
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
