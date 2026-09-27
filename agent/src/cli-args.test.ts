import { describe, expect, it } from "vitest";
import { CANONICAL_AGENT_JSONL, parseArgs, resolveRunTarget } from "./cli-args.js";

const KNOWN_ARMS = [
  "control-baseline",
  "control-oracle",
  "ratel-full",
  "ratel-pre-discovery",
  "ratel-discovery-tool",
];

const parse = (...argv: string[]) => parseArgs(argv, KNOWN_ARMS);

describe("parseArgs", () => {
  it("defaults", () => {
    const args = parse();
    expect(args).toMatchObject({
      corpus: "test-data/metatool.jsonl",
      output: "agent/results/agent.jsonl",
      outputExplicit: false,
      ephemeral: false,
      arms: KNOWN_ARMS,
      runs: 1,
      topK: 5,
      retriever: "bm25",
      poolSizes: [180],
      maxSteps: 12,
      timeoutMs: 60_000,
      force: false,
      noJudge: false,
      noAst: false,
      concurrency: 10,
      logLevel: "normal",
    });
    expect(args.cacheSources).toBeUndefined();
    expect(args.ratelVersion).toBeUndefined();
  });

  it("parses the campaign flags", () => {
    const args = parse(
      "--output",
      "out.jsonl",
      "--arms",
      "control-baseline, ratel-full",
      "--models",
      "claude-haiku-4-5,gpt-5.4-mini",
      "--pool-sizes",
      "100,30,100",
      "--retriever",
      "hybrid",
      "--no-judge",
      "--force",
      "-q",
    );
    expect(args).toMatchObject({
      output: "out.jsonl",
      outputExplicit: true,
      arms: ["control-baseline", "ratel-full"],
      models: ["claude-haiku-4-5", "gpt-5.4-mini"],
      poolSizes: [30, 100],
      retriever: "hybrid",
      noJudge: true,
      force: true,
      logLevel: "quiet",
    });
  });

  it("rejects unknown flags, unknown/renamed arms and bad pool sizes", () => {
    expect(() => parse("--nope")).toThrow("unknown flag: --nope");
    expect(() => parse("--arms", "ratel")).toThrow(/renamed to "ratel-full"/);
    expect(() => parse("--arms", "bogus")).toThrow(/unknown arm "bogus"/);
    expect(() => parse("--pool-size", "30,50")).toThrow(/Use --pool-sizes/);
    expect(() => parse("--concurrency", "0")).toThrow(/positive integer/);
    expect(() => parse("--output")).toThrow("missing value for --output");
  });

  it("--cache-source takes a comma list of paths", () => {
    expect(parse("--cache-source", "a.jsonl").cacheSources).toEqual(["a.jsonl"]);
    expect(parse("--cache-source", "a.jsonl, b.jsonl,").cacheSources).toEqual([
      "a.jsonl",
      "b.jsonl",
    ]);
    expect(() => parse("--cache-source", ",")).toThrow(/at least one path/);
  });

  it("--ratel-version rejected unless every --arms entry is a control arm", () => {
    const controls = ["--arms", "control-baseline,control-oracle"];
    expect(parse(...controls, "--ratel-version", "0.1.5").ratelVersion).toBe("0.1.5");
    // Flag order doesn't matter.
    expect(parse("--ratel-version", "0.1.5", ...controls).ratelVersion).toBe("0.1.5");

    expect(() =>
      parse("--arms", "control-baseline,ratel-full", "--ratel-version", "0.1.5"),
    ).toThrow(/--ratel-version.*control arms.*ratel-full/);
    // The default arm list includes the ratel arms.
    expect(() => parse("--ratel-version", "0.1.5")).toThrow(/--ratel-version/);
  });

  it("--ratel-version rejects an empty or flag-like value", () => {
    const controls = ["--arms", "control-baseline"];
    // e.g. `--ratel-version "$LABEL"` with LABEL unset, or a missing value eating the next flag.
    for (const v of ["", "  ", "--force"]) {
      expect(() => parse(...controls, "--ratel-version", v)).toThrow(/needs a version/);
    }
    expect(parse(...controls, "--ratel-version", " 0.1.5 ").ratelVersion).toBe("0.1.5");
  });
});

describe("resolveRunTarget", () => {
  const io = (existing: string[] = []) => ({
    resolve: (p: string) => `/repo/${p}`,
    exists: (p: string) => existing.includes(p),
    ephemeralOutput: () => "agent/results/ephemeral/agent-T.jsonl",
  });
  const controls = ["--arms", "control-baseline"];

  it("explicit --cache-source list resolves in order and beats every default", () => {
    const target = resolveRunTarget(
      parse("--ephemeral", "--cache-source", "b.jsonl,a.jsonl"),
      io(["/repo/agent/results/ephemeral/agent.jsonl"]),
    );
    expect(target.cacheSourcePaths).toEqual(["/repo/b.jsonl", "/repo/a.jsonl"]);
  });

  it("--ephemeral writes a fresh file and reads controls from the canonical agent.jsonl", () => {
    expect(resolveRunTarget(parse("--ephemeral"), io())).toEqual({
      outputPath: "/repo/agent/results/ephemeral/agent-T.jsonl",
      cacheSourcePaths: [`/repo/${CANONICAL_AGENT_JSONL}`],
      ratelVersion: undefined,
    });
  });

  it("--ephemeral with --output throws", () => {
    expect(() => resolveRunTarget(parse("--ephemeral", "--output", "x.jsonl"), io())).toThrow(
      /mutually exclusive/,
    );
  });

  it("defaults to the sibling agent.jsonl only when it exists and differs from the output", () => {
    const args = parse("--output", "results/agent-0.4.0-sparse.jsonl");
    expect(resolveRunTarget(args, io(["/repo/results/agent.jsonl"]))).toEqual({
      outputPath: "/repo/results/agent-0.4.0-sparse.jsonl",
      cacheSourcePaths: ["/repo/results/agent.jsonl"],
      ratelVersion: undefined,
    });
    expect(resolveRunTarget(args, io()).cacheSourcePaths).toBeUndefined();
    // Writing the canonical file itself: the runner falls back to the output.
    expect(
      resolveRunTarget(parse("--output", "results/agent.jsonl"), io(["/repo/results/agent.jsonl"]))
        .cacheSourcePaths,
    ).toBeUndefined();
  });

  it("passes --ratel-version through", () => {
    expect(
      resolveRunTarget(parse(...controls, "--ratel-version", "0.1.5"), io()).ratelVersion,
    ).toBe("0.1.5");
  });
});
