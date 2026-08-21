import { describe, expect, it } from "vitest";
import {
  catalogTokenEstimate,
  invokeSpans,
  metricsAtK,
  parseTelemetry,
  rankedHits,
  type SearchEvent,
  searchEvents,
  searchesUntilFirstGold,
  unionRanked,
  unionRecall,
  wastedSearches,
} from "./mcpatlas-gateway.js";
import { CODING_SERVERS } from "./mcpatlas-servers.js";
import type { RankedHit } from "./mcpatlas-types.js";

const SERVERS = [...CODING_SERVERS];

function search(hits: Array<[string, number]>, over: Partial<SearchEvent> = {}): SearchEvent {
  return {
    type: "search",
    query: "find repos",
    origin: "agent",
    top_k: 5,
    hits: hits.map(([tool_id, score]) => ({ tool_id, score })),
    stages: [{ name: "bm25", took_ms: 2, top_score: hits[0]?.[1] ?? 0 }],
    took_ms: 3,
    ...over,
  };
}

function hit(tool_id: string, rank: number, is_gold: boolean): RankedHit {
  return { rank, tool_id, score: 10 - rank, is_gold, server: tool_id.split("/")[0] };
}

describe("parseTelemetry", () => {
  it("reads a mixed stream", () => {
    const text = [
      JSON.stringify({ type: "ratel_tool_payload", server: "github", estimated_tokens: 100 }),
      JSON.stringify(search([["github__a", 1]])),
      JSON.stringify({ type: "invoke_start", tool_id: "github__a" }),
    ].join("\n");
    expect(parseTelemetry(text).map((e) => e.type)).toEqual([
      "ratel_tool_payload",
      "search",
      "invoke_start",
    ]);
  });

  it("skips truncated lines instead of throwing — the file is written live", () => {
    const text = `${JSON.stringify(search([["github__a", 1]]))}\n{"type":"sea`;
    expect(searchEvents(parseTelemetry(text))).toHaveLength(1);
  });

  it("returns empty for an empty file — callers must treat that as a ratel-cell error", () => {
    expect(parseTelemetry("")).toEqual([]);
    expect(parseTelemetry("\n \n")).toEqual([]);
  });
});

describe("catalogTokenEstimate", () => {
  it("sums the per-server registration estimates", () => {
    const events = parseTelemetry(
      [
        JSON.stringify({ type: "ratel_tool_payload", server: "github", estimated_tokens: 900 }),
        JSON.stringify({ type: "ratel_tool_payload", server: "git", estimated_tokens: 100 }),
        JSON.stringify(search([])),
      ].join("\n"),
    );
    expect(catalogTokenEstimate(events)).toBe(1000);
  });

  it("is 0 when no registration events were emitted", () => {
    expect(catalogTokenEstimate([])).toBe(0);
  });
});

describe("rankedHits — the score>0 filter", () => {
  const gold = ["github/search_repositories"];

  it("canonicalizes telemetry ids and flags gold", () => {
    const { ranked } = rankedHits(search([["github__search_repositories", 2.1]]), gold, SERVERS);
    expect(ranked[0]).toMatchObject({
      rank: 1,
      tool_id: "github/search_repositories",
      is_gold: true,
      server: "github",
    });
  });

  it("drops score:0 skill ride-alongs into a separate bucket", () => {
    const { ranked, zeroScore } = rankedHits(
      search([
        ["github__a", 2],
        ["git__b", 0],
        ["github__search_repositories", 1],
      ]),
      gold,
      SERVERS,
    );
    expect(ranked.map((h) => h.tool_id)).toEqual(["github/a", "github/search_repositories"]);
    expect(zeroScore).toEqual([{ tool_id: "git/b", score: 0 }]);
  });

  it("renumbers ranks AFTER filtering, so a ride-along cannot displace gold from top-k", () => {
    // Without filter-before-rank, gold sits at position 3 and hit@2 is false.
    const { ranked } = rankedHits(
      search([
        ["github__a", 2],
        ["git__ridealong", 0],
        ["github__search_repositories", 1],
      ]),
      gold,
      SERVERS,
    );
    expect(ranked.map((h) => h.rank)).toEqual([1, 2]);
    expect(metricsAtK(ranked, gold, 2).hit_at_k).toBe(true);
  });
});

describe("metricsAtK — mirrors retrieval/src/retrieval.rs", () => {
  const gold = ["a/1", "a/2"];
  const ranked = [hit("a/1", 1, true), hit("b/x", 2, false), hit("a/2", 3, true)];

  it("recall@k over the full gold set", () => {
    expect(metricsAtK(ranked, gold, 1).recall_at_k).toBe(0.5);
    expect(metricsAtK(ranked, gold, 3).recall_at_k).toBe(1);
  });

  it("precision@k divides by min(|ranked|, k), not k", () => {
    expect(metricsAtK(ranked, gold, 2).precision_at_k).toBe(0.5);
    // only 3 ranked, so k=5 divides by 3
    expect(metricsAtK(ranked, gold, 5).precision_at_k).toBeCloseTo(2 / 3, 10);
  });

  it("reciprocal rank is 1/first gold position, 0 when absent from top-k", () => {
    expect(metricsAtK(ranked, gold, 3).reciprocal_rank).toBe(1);
    expect(metricsAtK([hit("b/x", 1, false), hit("a/1", 2, true)], gold, 1).reciprocal_rank).toBe(
      0,
    );
    expect(metricsAtK([hit("b/x", 1, false), hit("a/1", 2, true)], gold, 2).reciprocal_rank).toBe(
      0.5,
    );
  });

  it("complete@k requires EVERY gold tool", () => {
    expect(metricsAtK(ranked, gold, 1).complete_at_k).toBe(false);
    expect(metricsAtK(ranked, gold, 3).complete_at_k).toBe(true);
  });

  it("nDCG is 1 for a perfect prefix and less when gold is buried", () => {
    expect(metricsAtK([hit("a/1", 1, true), hit("a/2", 2, true)], gold, 2).ndcg_at_k).toBeCloseTo(
      1,
      10,
    );
    expect(metricsAtK(ranked, gold, 3).ndcg_at_k).toBeLessThan(1);
  });

  it("best_gold_rank spans the WHOLE ranking, not just top-k", () => {
    const deep = [hit("b/x", 1, false), hit("b/y", 2, false), hit("a/1", 3, true)];
    const m = metricsAtK(deep, gold, 2);
    expect(m.hit_at_k).toBe(false);
    // the actionable part: gold existed at rank 3, so k=2 missed it by one
    expect(m.best_gold_rank).toBe(3);
  });

  it("is null-safe when gold never appears", () => {
    const m = metricsAtK([hit("b/x", 1, false)], gold, 5);
    expect(m.best_gold_rank).toBeNull();
    expect(m.gold_score).toBeNull();
    expect(m.hit_at_k).toBe(false);
  });

  it("handles an empty ranking without dividing by zero", () => {
    const m = metricsAtK([], gold, 3);
    expect(m).toMatchObject({
      recall_at_k: 0,
      precision_at_k: 0,
      reciprocal_rank: 0,
      ndcg_at_k: 0,
    });
  });
});

describe("multi-search aggregation", () => {
  const gold = ["a/1", "a/2"];
  const r1 = { ranked: [hit("a/1", 1, true), hit("b/x", 2, false)], zeroScore: [] };
  const r2 = { ranked: [hit("b/y", 1, false)], zeroScore: [] };
  const r3 = { ranked: [hit("a/2", 1, true)], zeroScore: [] };

  it("union dedupes by max score and re-ranks", () => {
    const u = unionRanked([r1, r3]);
    expect(u.map((h) => h.tool_id)).toEqual(["a/1", "a/2", "b/x"]);
    expect(u.map((h) => h.rank)).toEqual([1, 2, 3]);
  });

  it("union recall answers 'did the agent EVER see gold'", () => {
    expect(unionRecall([r1], gold)).toBe(0.5);
    expect(unionRecall([r1, r3], gold)).toBe(1);
    expect(unionRecall([r2], gold)).toBe(0);
  });

  it("counts searches until first gold, and null when never", () => {
    expect(searchesUntilFirstGold([r2, r1])).toBe(1);
    expect(searchesUntilFirstGold([r2])).toBeNull();
  });

  it("counts searches that returned no gold at all", () => {
    expect(wastedSearches([r1, r2, r3])).toBe(1);
  });
});

describe("invokeSpans", () => {
  it("pairs start with end and canonicalizes the id", () => {
    const events = parseTelemetry(
      [
        JSON.stringify({ type: "invoke_start", tool_id: "github__get_issue", args_size_bytes: 40 }),
        JSON.stringify({ type: "invoke_end", tool_id: "github__get_issue", took_ms: 120 }),
      ].join("\n"),
    );
    expect(invokeSpans(events, SERVERS)).toEqual([
      { tool_id: "github/get_issue", args_size_bytes: 40, took_ms: 120, error: null },
    ]);
  });

  it("surfaces invoke_error as a failed span", () => {
    const events = parseTelemetry(
      [
        JSON.stringify({ type: "invoke_start", tool_id: "git__status" }),
        JSON.stringify({ type: "invoke_error", tool_id: "git__status", error: "boom" }),
      ].join("\n"),
    );
    expect(invokeSpans(events, SERVERS)[0]).toMatchObject({ tool_id: "git/status", error: "boom" });
  });

  it("still reports an invoke that never terminated", () => {
    const events = parseTelemetry(JSON.stringify({ type: "invoke_start", tool_id: "git__status" }));
    expect(invokeSpans(events, SERVERS)[0]).toMatchObject({ error: "no invoke_end" });
  });
});
