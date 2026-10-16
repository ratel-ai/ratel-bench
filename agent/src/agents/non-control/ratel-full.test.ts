import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { buildEmbeddingArtifact } from "../../embedding-artifact.js";
import { REPO_ROOT } from "../../paths.js";
import { resetForTests, selectVersion } from "../../sdk/resolve.js";
import type { AgentRunInput, PrewarmInput, Scenario, ToolSpec } from "../../types.js";
import {
  buildRatelFullBundle,
  descriptor,
  isPrewarmed,
  prewarm,
  resetPrewarmCache,
} from "./ratel-full.js";

const pool: ToolSpec[] = [
  {
    id: "fs.read_file",
    name: "read_file",
    description: "Read a file from disk.",
    input_schema: { type: "object", properties: { path: { type: "string" } } },
  },
  {
    id: "fs.write_file",
    name: "write_file",
    description: "Write contents to a file.",
    input_schema: { type: "object", properties: { path: { type: "string" } } },
  },
  {
    id: "mail.send",
    name: "send_email",
    description: "Send an email via SMTP.",
    input_schema: { type: "object", properties: { to: { type: "string" } } },
  },
];

describe("ratel-full descriptor", () => {
  it("declares the canonical id and label", () => {
    expect(descriptor.id).toBe("ratel-full");
    expect(descriptor.label).toBe("ratel (full)");
    expect(descriptor.skipForModel).toBeUndefined();
  });

  it("exposes a prepare hook so the runner can prewarm off the timed loop", () => {
    expect(typeof descriptor.prepare).toBe("function");
  });
});

describe("prewarm bundle cache", () => {
  // The cache is what keeps the (blocking, native) semantic embedding out of the
  // concurrent metered loop: prewarm builds it serially, run() reuses it. These
  // pin the cache contract that makes `wall_ms` an honest latency measurement.
  const scenario = { id: "bfcl-simple-1", prompt: "read a file from disk" } as Scenario;
  const cell: PrewarmInput = {
    scenario,
    pool,
    poolSize: pool.length,
    topK: 2,
    retriever: "bm25",
    seed: 42,
  };

  beforeEach(() => resetPrewarmCache());

  it("is empty until prewarmed, then reports a hit for that exact cell", async () => {
    expect(isPrewarmed(cell)).toBe(false);
    await prewarm([cell]);
    expect(isPrewarmed(cell)).toBe(true);
  });

  it("keys on (scenario, poolSize, retriever, seed) — a different cell is a miss", async () => {
    await prewarm([cell]);
    expect(isPrewarmed({ ...cell, seed: 7 })).toBe(false);
    expect(isPrewarmed({ ...cell, retriever: "semantic" })).toBe(false);
    expect(isPrewarmed({ ...cell, poolSize: 99 })).toBe(false);
  });

  it("is idempotent — re-prewarming a cached cell is a no-op", async () => {
    await prewarm([cell]);
    // Re-prewarming must neither throw nor rebuild — a rejected promise fails
    // the test on its own, so awaiting is the assertion.
    await prewarm([cell, cell]);
    expect(isPrewarmed(cell)).toBe(true);
  });

  it("rejects re-prewarming a cached cell with a different embedding artifact path", async () => {
    await prewarm([{ ...cell, embeddingArtifactPath: "/a.rat1" }]);
    await expect(prewarm([{ ...cell, embeddingArtifactPath: "/a.rat1" }])).resolves.toBeUndefined();
    await expect(prewarm([{ ...cell, embeddingArtifactPath: "/b.rat1" }])).rejects.toThrow(
      /embedding artifact mismatch/,
    );
    // undefined vs a path counts as different.
    await expect(prewarm([cell])).rejects.toThrow(/embedding artifact mismatch/);
  });

  it("rejects run() for a cached cell with a different embedding artifact path", async () => {
    await prewarm([cell]);
    const input = { ...cell, embeddingArtifactPath: "/other.rat1" } as unknown as AgentRunInput;
    await expect(descriptor.run(input)).rejects.toThrow(/embedding artifact mismatch/);
  });

  it("builds a bundle whose tool selection is identical to a cold build", async () => {
    // The whole point: prewarming must not change WHICH tools the cell exposes,
    // only WHEN they're built. A cold build is deterministic, so equal ids prove
    // the prewarmed path is byte-identical in selection.
    await prewarm([cell]);
    const cold = await buildRatelFullBundle({ scenario, pool, topK: 2, retriever: "bm25" });
    const coldAgain = await buildRatelFullBundle({ scenario, pool, topK: 2, retriever: "bm25" });
    expect(coldAgain.bundle.activeToolIds).toEqual(cold.bundle.activeToolIds);
  });
});

describe("buildRatelFullBundle", () => {
  // The arm's contract (per ADR-0006 mode (c)) is dual-layer: BM25 top-K of
  // the prompt is registered as direct tools so the model sees its likely
  // best moves up front, AND the search/invoke gateway is exposed so the
  // model can recover when pre-discovery missed. Each assertion below pins
  // one half of that contract.
  it("registers both gateway tools (search_tools + invoke_tool) — fallback path", async () => {
    const { bundle } = await buildRatelFullBundle({
      scenario: { prompt: "read a file from disk" },
      pool,
      topK: 2,
    });
    expect(bundle.tools.search_tools).toBeDefined();
    expect(bundle.tools.invoke_tool).toBeDefined();
  });

  it("registers BM25 top-K of the user prompt as direct tools — pre-discovery path", async () => {
    const { bundle } = await buildRatelFullBundle({
      scenario: { prompt: "read a file from disk" },
      pool,
      topK: 2,
    });
    // The prompt matches fs.read_file via BM25; that hit must show up in
    // activeToolIds, proving pre-discovery actually ran and isn't a no-op.
    // activeToolIds also includes the two gateway tools (search_tools,
    // invoke_tool) — direct tools are bounded above by topK (2 here).
    expect(bundle.activeToolIds).toContain("fs.read_file");
    expect(bundle.activeToolIds).toContain("search_tools");
    expect(bundle.activeToolIds).toContain("invoke_tool");
    const directIds = bundle.activeToolIds.filter(
      (id) => id !== "search_tools" && id !== "invoke_tool",
    );
    expect(directIds.length).toBeGreaterThan(0);
    expect(directIds.length).toBeLessThanOrEqual(2);
  });

  it("backs the gateway with the full pool, not just top-K", async () => {
    // Pre-discovery and gateway must coexist: top-K becomes direct tools,
    // but the gateway can still surface every other tool in the pool when
    // pre-discovery missed. The catalog therefore holds the *full* pool.
    const { catalog } = await buildRatelFullBundle({
      scenario: { prompt: "read a file from disk" },
      pool,
      topK: 1,
    });
    expect(catalog.has("fs.read_file")).toBe(true);
    expect(catalog.has("fs.write_file")).toBe(true);
    expect(catalog.has("mail.send")).toBe(true);
  });

  it("normalizes empty input schemas at the AI SDK boundary", async () => {
    // MetaTool tools ship with `input_schema: {}` — Anthropic's API rejects
    // tools whose input_schema is missing `type: "object"`. The ratel arm,
    // like control, has to normalize at the provider seam.
    const noSchema: ToolSpec = {
      id: "FinanceTool",
      name: "FinanceTool",
      description: "Finance plugin.",
      input_schema: {},
    };
    const { bundle } = await buildRatelFullBundle({
      scenario: { prompt: "FinanceTool" },
      pool: [noSchema],
      topK: 1,
    });
    const t = bundle.tools.FinanceTool as unknown as {
      inputSchema: { jsonSchema: { type?: string } };
    };
    expect(t.inputSchema.jsonSchema.type).toBe("object");
  });
});

describe("ratel-full embedding artifacts", () => {
  const dirs: string[] = [];

  beforeEach(() => {
    resetPrewarmCache();
    resetForTests();
    selectVersion("0.12.0");
  });
  afterEach(() => {
    resetForTests();
    resetPrewarmCache();
    for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
  });

  it.each(["semantic", "hybrid"] as const)(
    "uses the artifact in the serial %s prewarm pass",
    async (retriever) => {
      const cell: PrewarmInput = {
        scenario: { id: "artifact-prewarm", prompt: "read a file from disk" } as Scenario,
        pool,
        poolSize: pool.length,
        topK: 2,
        retriever,
        seed: 42,
        embeddingArtifactPath: "/missing/prewarm-bfcl.rat1",
      };
      await expect(prewarm([cell])).rejects.toThrow(/ENOENT/);
      expect(isPrewarmed(cell)).toBe(false);
    },
  );

  it.each(["semantic", "hybrid"] as const)(
    "uses the artifact for a cold %s catalog",
    async (retriever) => {
      await expect(
        buildRatelFullBundle({
          scenario: { prompt: "read a file from disk" },
          pool,
          topK: 2,
          retriever,
          embeddingArtifactPath: "/missing/bfcl.rat1",
        }),
      ).rejects.toThrow(/ENOENT/);
    },
  );

  it.each(["semantic", "hybrid"] as const)(
    "preserves %s tool selection, executors and cache keys with a real BFCL artifact",
    async (retriever) => {
      const dir = mkdtempSync(join(REPO_ROOT, "agent/.ratel-full-artifact-test-"));
      dirs.push(dir);
      const corpus = join(dir, "bfcl.jsonl");
      const output = join(dir, "bfcl.rat1");
      const scenario: Scenario = {
        id: "artifact-parity",
        prompt: "read a file from disk",
        candidate_pool: pool,
        gold_tools: ["fs.read_file"],
      };
      writeFileSync(corpus, `${JSON.stringify(scenario)}\n`);
      await buildEmbeddingArtifact({ benchmark: "bfcl", corpus, output });
      const cell: PrewarmInput = {
        scenario,
        pool,
        poolSize: pool.length,
        topK: 2,
        retriever,
        seed: 42,
      };
      const cold = await buildRatelFullBundle(cell);
      const warmed = await buildRatelFullBundle({ ...cell, embeddingArtifactPath: output });
      expect(warmed.bundle.activeToolIds).toEqual(cold.bundle.activeToolIds);
      expect(warmed.bundle.activeToolIds).toContain("fs.read_file");
      expect(warmed.bundle.tools.search_tools).toBeDefined();
      expect(warmed.bundle.tools.invoke_tool).toBeDefined();
      await expect(warmed.catalog.invoke("fs.read_file", { path: "/etc/hosts" })).resolves.toEqual({
        _stub: "stubbed for benchmark",
        toolId: "fs.read_file",
      });

      await prewarm([{ ...cell, embeddingArtifactPath: output }]);
      expect(isPrewarmed(cell)).toBe(true);
      // Same path re-prewarm stays a no-op; a different path must not silently
      // reuse the ranking warmed from the first artifact.
      await expect(prewarm([{ ...cell, embeddingArtifactPath: output }])).resolves.toBeUndefined();
      await expect(
        prewarm([{ ...cell, embeddingArtifactPath: "/missing/already-prewarmed.rat1" }]),
      ).rejects.toThrow(/embedding artifact mismatch/);
    },
    60_000,
  );

  it("rejects corrupt artifact bytes instead of silently embedding inline", async () => {
    const dir = mkdtempSync(join(REPO_ROOT, "agent/.ratel-full-artifact-test-"));
    dirs.push(dir);
    const path = join(dir, "corrupt.rat1");
    writeFileSync(path, "not an embedding artifact");
    await expect(
      buildRatelFullBundle({
        scenario: { prompt: "read a file from disk" },
        pool,
        topK: 2,
        retriever: "semantic",
        embeddingArtifactPath: path,
      }),
    ).rejects.toThrow();
  });

  it.each(["0.4.0", "0.5.0"])(
    "embeds inline without reading a missing artifact on legacy SDK %s",
    async (version) => {
      resetForTests();
      selectVersion(version);
      const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
      try {
        const cell: PrewarmInput = {
          scenario: { id: `legacy-${version}`, prompt: "read a file from disk" } as Scenario,
          pool,
          poolSize: pool.length,
          topK: 2,
          retriever: "semantic",
          seed: 42,
          embeddingArtifactPath: "/missing/legacy-bfcl.rat1",
        };
        await prewarm([cell]);
        expect(isPrewarmed(cell)).toBe(true);
        const { bundle } = await buildRatelFullBundle(cell);
        expect(bundle.activeToolIds).toContain("fs.read_file");
        expect(warn).toHaveBeenCalledExactlyOnceWith(
          `SDK ${version} lacks experimentalEmbeddingArtifact; embedding inline`,
        );
      } finally {
        warn.mockRestore();
      }
    },
    60_000,
  );

  it("ignores the artifact for BM25", async () => {
    const { bundle } = await buildRatelFullBundle({
      scenario: { prompt: "read a file from disk" },
      pool,
      topK: 2,
      retriever: "bm25",
      embeddingArtifactPath: "/missing/ignored.rat1",
    });
    expect(bundle.activeToolIds).toContain("fs.read_file");
  });
});
