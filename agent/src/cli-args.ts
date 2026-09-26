// Argument parsing for the campaign CLI (`pnpm start`). Kept apart from
// `cli.ts`, which runs `main()` on import, so the parser is unit-testable.

import { dirname, join } from "node:path";
import type { JudgePromptVariant } from "./judges/llm.js";
import type { OutputCapOverride } from "./output-limits.js";
import { CACHEABLE_ARMS, type RunnerConfig } from "./runner.js";
import type { Arm, RetrievalMethod } from "./types.js";

export const DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1";

/** Canonical agent.jsonl that ephemeral runs read for cached control rows. */
export const CANONICAL_AGENT_JSONL = "agent/results/agent.jsonl";

/** Default arms when `--arms` isn't passed: every committed arm. The local-only
 * `claude-sdk-tool-search` is included automatically by the registry but
 * excluded from the default list — opt in via `--arms` once it's wired locally. */
const DEFAULT_ARMS: Arm[] = [
  "control-baseline",
  "control-oracle",
  "ratel-full",
  "ratel-pre-discovery",
  "ratel-discovery-tool",
];

/** Old → new id hints for the rename in v0.1.2. Pre-empts a confusing
 * `unknown arm` error when developers re-run an older command. */
const RENAMES: Record<string, string> = {
  control: "control-baseline",
  oracle: "control-oracle",
  ratel: "ratel-full",
  hybrid: "ratel-full",
};

export interface ParsedArgs {
  corpus: string;
  output: string;
  outputExplicit: boolean;
  ephemeral: boolean;
  /** Reuse version-independent control cells (baseline/oracle) from these files
   * (`--cache-source a,b,…`; exact harness tier first, then earliest) instead of
   * re-running them. Lets a per-method output file still pull cached controls. */
  cacheSources?: string[];
  /** `--ratel-version V`: stamp (and resume) rows as version V instead of the
   * installed SDK's. Controls only — used to re-drain a label's controls. */
  ratelVersion?: string;
  scenarios?: number;
  arms: Arm[];
  models: string[];
  runs: number;
  topK: number;
  /** Retrieval method for the Ratel arms (bm25 | semantic | hybrid). Defaults to bm25. */
  retriever: RetrievalMethod;
  poolSizes: number[];
  maxSteps: number;
  timeoutMs: number;
  dollarGlobal: number;
  force: boolean;
  noJudge: boolean;
  /** Skip the (LLM-free) argument-level task-completion verdict. Defaults to off. */
  noAst: boolean;
  /** Override the LLM judge model. Defaults to claude-sonnet-4-6 if ANTHROPIC_API_KEY is set. */
  judgeModelId?: string;
  /**
   * `--max-output-tokens N|none`: per-call cap for every agent model, overriding
   * models.json `maxOutputTokens` (`none` = send no cap). Unset = the catalog's.
   */
  maxOutputTokens?: OutputCapOverride;
  /** `--judge-max-output-tokens N`: cap on each LLM-judge call. Unset = no cap sent. */
  judgeMaxOutputTokens?: number;
  ollamaBaseURL: string;
  /** Optional bearer token for a user-hosted (`<url>#<model>`) endpoint. */
  modelApiKey?: string;
  seed: number;
  /** Cells in flight at once. See `RunnerConfig.concurrency` for cap semantics. */
  concurrency: number;
  logLevel: "quiet" | "normal" | "verbose";
}

/**
 * Where a run writes, which files feed its control cache (and which cache tiers
 * may serve), the version it's filed under, and the judge's output cap.
 */
export type RunTarget = Pick<
  RunnerConfig,
  "outputPath" | "cacheSourcePaths" | "ratelVersion" | "allowLegacyCache" | "judgeMaxOutputTokens"
>;

/** Filesystem access `resolveRunTarget` needs, injected so it stays pure. */
export interface RunTargetIO {
  /** Resolve a CLI path against the repo root. */
  resolve: (path: string) => string;
  exists: (path: string) => boolean;
  /** Fresh per-run output path for `--ephemeral`. */
  ephemeralOutput: () => string;
}

export function parseArgs(argv: string[], knownArms: readonly string[]): ParsedArgs {
  const args: ParsedArgs = {
    corpus: "test-data/metatool.jsonl",
    output: "agent/results/agent.jsonl",
    outputExplicit: false,
    ephemeral: false,
    arms: [...DEFAULT_ARMS],
    models: ["gpt-5.4-mini", "claude-sonnet-4-6"],
    runs: 1,
    topK: 5,
    retriever: "bm25",
    poolSizes: [180],
    maxSteps: 12,
    timeoutMs: 60_000,
    dollarGlobal: 25,
    force: false,
    noJudge: false,
    noAst: false,
    ollamaBaseURL: process.env.OLLAMA_BASE_URL ?? DEFAULT_OLLAMA_BASE_URL,
    modelApiKey: process.env.AWS_BEDROCK_BEARER,
    seed: 42,
    concurrency: 10,
    logLevel: "normal",
  };
  for (let i = 0; i < argv.length; i++) {
    const flag = argv[i];
    const next = (): string => {
      const v = argv[++i];
      if (v === undefined) throw new Error(`missing value for ${flag}`);
      return v;
    };
    switch (flag) {
      case "--corpus":
        args.corpus = next();
        break;
      case "--output":
        args.output = next();
        args.outputExplicit = true;
        break;
      case "--ephemeral":
        args.ephemeral = true;
        break;
      case "--cache-source":
        args.cacheSources = parseCacheSources(next());
        break;
      case "--ratel-version": {
        // Catch an unset `$LABEL` ("") or a missing value swallowing the next flag,
        // which would otherwise stamp every row with a bogus version.
        const v = next().trim();
        if (!v || v.startsWith("-")) {
          throw new Error(`--ratel-version needs a version, e.g. 0.1.5 (got "${v}")`);
        }
        args.ratelVersion = v;
        break;
      }
      case "--scenarios":
        args.scenarios = Number(next());
        break;
      case "--arms":
        args.arms = parseArms(next(), knownArms);
        break;
      case "--models":
        args.models = next().split(",");
        break;
      case "--runs":
        args.runs = Number(next());
        break;
      case "--top-k":
        args.topK = Number(next());
        break;
      case "--retriever": {
        const v = next();
        if (v !== "bm25" && v !== "semantic" && v !== "hybrid") {
          throw new Error(`--retriever must be bm25, semantic, or hybrid (got "${v}")`);
        }
        args.retriever = v;
        break;
      }
      case "--pool-size":
        args.poolSizes = [parsePoolSize(flag, next())];
        break;
      case "--pool-sizes":
        args.poolSizes = parsePoolSizes(next());
        break;
      case "--max-steps":
        args.maxSteps = Number(next());
        break;
      case "--timeout-ms":
        args.timeoutMs = Number(next());
        break;
      case "--dollar-global":
        args.dollarGlobal = Number(next());
        break;
      case "--force":
        args.force = true;
        break;
      case "--no-judge":
        args.noJudge = true;
        break;
      case "--no-ast":
        args.noAst = true;
        break;
      case "--judge-model":
        args.judgeModelId = next();
        break;
      case "--max-output-tokens":
        args.maxOutputTokens = parseOutputCapFlag(flag, next());
        break;
      case "--judge-max-output-tokens":
        args.judgeMaxOutputTokens = parsePositiveInt(flag, next());
        break;
      case "--ollama-base-url":
        args.ollamaBaseURL = next();
        break;
      case "--model-api-key":
        args.modelApiKey = next();
        break;
      case "--seed":
        args.seed = Number(next());
        break;
      case "--concurrency": {
        const n = Number(next());
        if (!Number.isFinite(n) || n < 1 || !Number.isInteger(n)) {
          throw new Error(`--concurrency must be a positive integer, got ${n}`);
        }
        args.concurrency = n;
        break;
      }
      case "--verbose":
      case "-v":
        args.logLevel = "verbose";
        break;
      case "--quiet":
      case "-q":
        args.logLevel = "quiet";
        break;
      default:
        throw new Error(`unknown flag: ${flag}`);
    }
  }
  // Validated after the loop: --ratel-version and --arms may come in any order.
  if (args.ratelVersion !== undefined) assertControlArmsOnly(args.arms);
  return args;
}

/**
 * Resolve the run's output and control-cache sources. Control reuse is ON by
 * default; sources, in precedence order:
 *  1. `--cache-source a,b,…`;
 *  2. `--ephemeral`: the canonical `agent/results/agent.jsonl`;
 *  3. the sibling `agent.jsonl` in the output's directory, when it exists and
 *     isn't the output itself (e.g. a per-method `agent-0.4.0-sparse.jsonl`);
 *  4. otherwise `undefined`: the runner falls back to the output's own rows.
 * A model with no cached controls just runs them live; `--force` disables reuse.
 * An explicit `--max-output-tokens` (N or `none`) must measure the cap it names,
 * so it serves exact-tier controls only (`allowLegacyCache: false`).
 */
export function resolveRunTarget(parsed: ParsedArgs, io: RunTargetIO): RunTarget {
  if (parsed.ephemeral && parsed.outputExplicit) {
    throw new Error("--ephemeral and --output are mutually exclusive");
  }
  const output = parsed.ephemeral ? io.ephemeralOutput() : parsed.output;
  const outputPath = io.resolve(output);
  return {
    outputPath,
    cacheSourcePaths: cacheSourcesFor(parsed, output, outputPath, io),
    ratelVersion: parsed.ratelVersion,
    allowLegacyCache: parsed.maxOutputTokens === undefined,
    judgeMaxOutputTokens: parsed.judgeMaxOutputTokens,
  };
}

export interface RejudgeParsedArgs {
  input: string;
  corpus: string;
  judgeModelId?: string;
  promptVariant: JudgePromptVariant;
  out?: string;
  ollamaBaseURL: string;
  /** Bearer token for a user-hosted (`<url>#<model>`) judge endpoint (optional). */
  modelApiKey?: string;
  /** Skip the LLM judge — only recompute the (LLM-free) AST task-completion verdict. */
  noJudge: boolean;
  /** `--judge-max-output-tokens N`: cap on each judge call. Unset = no cap sent. */
  judgeMaxOutputTokens?: number;
}

/** Parse `pnpm start rejudge <results.jsonl> [flags]` (argv after `rejudge`). */
export function parseRejudgeArgs(argv: string[]): RejudgeParsedArgs {
  const args: RejudgeParsedArgs = {
    input: "",
    corpus: "test-data/metatool.jsonl",
    promptVariant: "strict",
    ollamaBaseURL: process.env.OLLAMA_BASE_URL ?? DEFAULT_OLLAMA_BASE_URL,
    modelApiKey: process.env.AWS_BEDROCK_BEARER,
    noJudge: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const flag = argv[i];
    const next = (): string => {
      const v = argv[++i];
      if (v === undefined) throw new Error(`missing value for ${flag}`);
      return v;
    };
    switch (flag) {
      case "--corpus":
        args.corpus = next();
        break;
      case "--judge-model":
        args.judgeModelId = next();
        break;
      case "--judge-prompt": {
        const v = next();
        if (v !== "coherence" && v !== "strict") {
          throw new Error(`--judge-prompt must be "coherence" or "strict", got "${v}"`);
        }
        args.promptVariant = v;
        break;
      }
      case "--out":
        args.out = next();
        break;
      case "--no-judge":
        args.noJudge = true;
        break;
      case "--judge-max-output-tokens":
        args.judgeMaxOutputTokens = parsePositiveInt(flag, next());
        break;
      case "--ollama-base-url":
        args.ollamaBaseURL = next();
        break;
      case "--model-api-key":
        args.modelApiKey = next();
        break;
      default:
        if (flag.startsWith("-")) {
          throw new Error(`unknown flag for rejudge: ${flag}`);
        }
        if (args.input) {
          throw new Error(`rejudge takes a single input JSONL (got "${args.input}" and "${flag}")`);
        }
        args.input = flag;
    }
  }
  if (!args.input) {
    throw new Error(
      "rejudge: missing input JSONL. Usage:\n" +
        "  pnpm start rejudge <results.jsonl> [--corpus PATH] [--judge-model ID] " +
        "[--judge-prompt coherence|strict] [--judge-max-output-tokens N] [--out PATH]",
    );
  }
  return args;
}

/**
 * Parse a flag value that must be a positive integer (digits only, ≥ 1). Shared
 * by every integer-valued flag that must not silently become `NaN`/`0`.
 */
export function parsePositiveInt(flag: string, raw: string): number {
  const n = Number(raw);
  if (!/^\s*\d+\s*$/.test(raw) || !Number.isSafeInteger(n) || n < 1) {
    throw new Error(`${flag} must be a positive integer (got "${raw}")`);
  }
  return n;
}

/** `--max-output-tokens N|none`: a positive integer, or `none` to send no cap. */
export function parseOutputCapFlag(flag: string, raw: string): OutputCapOverride {
  if (raw.trim() === "none") return "none";
  try {
    return parsePositiveInt(flag, raw);
  } catch {
    throw new Error(`${flag} must be a positive integer or "none" (got "${raw}")`);
  }
}

/**
 * Parse + validate the `--arms` value against the registry. Bad input used to
 * flow through `as Arm[]` and crash deep in the runner with a useless
 * TypeError; this surface validates at the boundary and surfaces both the
 * legacy → new id rename and the full set of known ids.
 */
function parseArms(raw: string, knownArms: readonly string[]): Arm[] {
  const parts = raw
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
  if (parts.length === 0) throw new Error("--arms must list at least one arm");
  const out: Arm[] = [];
  for (const p of parts) {
    if (knownArms.includes(p)) {
      out.push(p);
      continue;
    }
    if (RENAMES[p]) {
      throw new Error(
        `--arms: "${p}" was renamed to "${RENAMES[p]}". Update your command to ` +
          `--arms ${DEFAULT_ARMS.join(",")} (or whichever subset you want).`,
      );
    }
    throw new Error(`--arms: unknown arm "${p}" (expected one of: ${knownArms.join(", ")})`);
  }
  return out;
}

/**
 * Parse a single positive integer for `--pool-size`. Rejects commas explicitly
 * so a `--pool-size 30,50,100` typo points the user at `--pool-sizes`
 * instead of silently flowing `NaN` through to `expandPool` (which would
 * collapse the catalog to gold-only).
 */
function parsePoolSize(flag: string, raw: string): number {
  if (raw.includes(",")) {
    throw new Error(
      `${flag} takes a single integer (got "${raw}"). Use --pool-sizes for a comma-separated sweep.`,
    );
  }
  return parsePositiveInt(flag, raw);
}

/** Parse `--pool-sizes 30,50,100` into a deduped, sorted list of positive integers. */
function parsePoolSizes(raw: string): number[] {
  const parts = raw
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
  if (parts.length === 0) throw new Error("--pool-sizes must list at least one integer");
  const seen = new Set<number>();
  for (const p of parts) {
    const n = Number(p);
    if (!Number.isFinite(n) || !Number.isInteger(n) || n < 1) {
      throw new Error(`--pool-sizes: "${p}" is not a positive integer`);
    }
    seen.add(n);
  }
  return [...seen].sort((a, b) => a - b);
}

/** Parse `--cache-source a,b,…` into a non-empty list of paths. */
function parseCacheSources(raw: string): string[] {
  const paths = raw
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
  if (paths.length === 0) throw new Error("--cache-source must list at least one path");
  return paths;
}

/**
 * `--ratel-version` re-stamps rows with a version the installed SDK didn't
 * produce. Only control arms are version-independent, so a Ratel arm run
 * under a borrowed version would be mislabelled.
 */
function assertControlArmsOnly(arms: readonly Arm[]): void {
  const nonControl = arms.filter((arm) => !CACHEABLE_ARMS.has(arm));
  if (nonControl.length > 0) {
    throw new Error(
      `--ratel-version applies to control arms only (${[...CACHEABLE_ARMS].join(", ")}); ` +
        `--arms also lists: ${nonControl.join(", ")}`,
    );
  }
}

function cacheSourcesFor(
  parsed: ParsedArgs,
  output: string,
  outputPath: string,
  io: RunTargetIO,
): string[] | undefined {
  if (parsed.cacheSources) return parsed.cacheSources.map((p) => io.resolve(p));
  if (parsed.ephemeral) return [io.resolve(CANONICAL_AGENT_JSONL)];
  const sibling = io.resolve(join(dirname(output), "agent.jsonl"));
  return sibling !== outputPath && io.exists(sibling) ? [sibling] : undefined;
}
