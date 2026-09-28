import { describe, expect, it } from "vitest";
import {
  CANONICAL_AGENT_JSONL,
  parseArgs,
  parseOutputCapFlag,
  parseRejudgeArgs,
  resolveRunTarget,
} from "./cli-args.js";

const KNOWN_ARMS = [
  "control-baseline",
  "control-oracle",
  "ratel-full",
  "ratel-pre-discovery",
  "ratel-discovery-tool",
];

const parse = (...argv: string[]) => parseArgs(argv, KNOWN_ARMS);

describe("parseArgs", () => {
  it("accepts an exact SDK release independently of the control-row label", () => {
    expect(parse("--sdk-version", "0.12.0").sdkVersion).toBe("0.12.0");
    expect(() => parse("--sdk-version", "")).toThrow(/--sdk-version/);
  });
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
    // No flag defaults: models.json supplies the agent cap, the judge gets none.
    expect(args.maxOutputTokens).toBeUndefined();
    expect(args.judgeMaxOutputTokens).toBeUndefined();
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

describe("output cap flags", () => {
  it("--max-output-tokens takes N or 'none'; --judge-max-output-tokens takes N", () => {
    expect(parse("--max-output-tokens", "16").maxOutputTokens).toBe(16);
    expect(parse("--max-output-tokens", "none").maxOutputTokens).toBe("none");
    expect(parse("--judge-max-output-tokens", "512").judgeMaxOutputTokens).toBe(512);
  });

  it("--timeout-ms (the active-time deadline) rejects non-positive-int values", () => {
    expect(parse("--timeout-ms", "180000").timeoutMs).toBe(180_000);
    for (const bad of ["0", "-1", "1.5", "abc", "1e3"]) {
      expect(() => parse("--timeout-ms", bad)).toThrow(/--timeout-ms must be a positive integer/);
    }
    expect(() => parse("--timeout-ms", "2147483648")).toThrow(/--timeout-ms must be ≤ 2147483647/);
  });

  it("cap flags reject non-positive-int values", () => {
    // "1e3", "+5", "0x10", "1.0" pass Number()/isInteger; only the digits-only rule rejects them.
    for (const bad of ["0", "-1", "1.5", "abc", "", "16,32", "1e3x", "1e3", "+5", "0x10", "1.0"]) {
      expect(() => parse("--max-output-tokens", bad)).toThrow(
        /--max-output-tokens must be a positive integer/,
      );
      expect(() => parse("--judge-max-output-tokens", bad)).toThrow(
        /--judge-max-output-tokens must be a positive integer/,
      );
    }
    // The judge cap has no 'none': unset already means no cap.
    expect(() => parse("--judge-max-output-tokens", "none")).toThrow(/positive integer/);
    expect(() => parse("--max-output-tokens")).toThrow("missing value for --max-output-tokens");
  });

  it("parseOutputCapFlag", () => {
    expect(parseOutputCapFlag("--x", "none")).toBe("none");
    expect(parseOutputCapFlag("--x", "42")).toBe(42);
  });

  it("[guard] --pool-size keeps its messages after delegating to parsePositiveInt", () => {
    expect(() => parse("--pool-size", "30,50")).toThrow(/Use --pool-sizes/);
    expect(() => parse("--pool-size", "0")).toThrow(
      '--pool-size must be a positive integer (got "0")',
    );
  });
});

describe("rerun flags", () => {
  it("default to --retry-errors infra, --max-attempts 3, --retry-rounds 1, --retry-delay-s 60", () => {
    expect(parse().rerun).toEqual({ policy: "infra", maxAttempts: 3, rounds: 1, delayMs: 60_000 });
  });

  it("parse --retry-errors, --max-attempts (0 = unlimited), --retry-rounds, --retry-delay-s", () => {
    const args = parse(
      "--retry-errors",
      "all",
      "--max-attempts",
      "0",
      "--retry-rounds",
      "0",
      "--retry-delay-s",
      "5",
    );
    expect(args.rerun).toEqual({ policy: "all", maxAttempts: 0, rounds: 0, delayMs: 5_000 });
  });

  it("reject bad values", () => {
    expect(() => parse("--retry-errors", "transient")).toThrow(
      '--retry-errors must be infra, all or none (got "transient")',
    );
    for (const flag of ["--max-attempts", "--retry-rounds", "--retry-delay-s"]) {
      for (const bad of ["-1", "1.5", "x"]) {
        expect(() => parse(flag, bad)).toThrow(`${flag} must be a non-negative integer`);
      }
      expect(() => parse(flag)).toThrow(`missing value for ${flag}`);
    }
    // The delay becomes a timer: at most 2147483 s.
    expect(() => parse("--retry-delay-s", "2147484")).toThrow(/--retry-delay-s must be ≤ 2147483/);
  });
});

describe("parseRejudgeArgs", () => {
  it("parses --judge-max-output-tokens N; unset by default", () => {
    expect(
      parseRejudgeArgs(["in.jsonl", "--judge-max-output-tokens", "256"]).judgeMaxOutputTokens,
    ).toBe(256);
    expect(parseRejudgeArgs(["in.jsonl"]).judgeMaxOutputTokens).toBeUndefined();
  });

  it("rejects a non-positive-int or missing --judge-max-output-tokens", () => {
    for (const bad of ["0", "abc", "none", "1e3"]) {
      expect(() => parseRejudgeArgs(["in.jsonl", "--judge-max-output-tokens", bad])).toThrow(
        /--judge-max-output-tokens must be a positive integer/,
      );
    }
    expect(() => parseRejudgeArgs(["in.jsonl", "--judge-max-output-tokens"])).toThrow(
      "missing value for --judge-max-output-tokens",
    );
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
      allowLegacyCache: true,
      judgeMaxOutputTokens: undefined,
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
      allowLegacyCache: true,
      judgeMaxOutputTokens: undefined,
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

  it("an explicit --max-output-tokens (N or none) serves no legacy-tier controls", () => {
    expect(resolveRunTarget(parse(), io()).allowLegacyCache).toBe(true);
    for (const cap of ["16", "none"]) {
      expect(resolveRunTarget(parse("--max-output-tokens", cap), io()).allowLegacyCache).toBe(
        false,
      );
    }
  });

  it("carries the judge cap alongside the agent cap", () => {
    const target = resolveRunTarget(
      parse("--max-output-tokens", "16", "--judge-max-output-tokens", "256"),
      io(),
    );
    expect(target).toMatchObject({ allowLegacyCache: false, judgeMaxOutputTokens: 256 });
  });
});
