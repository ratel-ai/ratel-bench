// CLI entry. Wires AI SDK provider models to the runner. Default scenario
// corpus is the ingested MetaTool snapshot, which `pnpm -F @ratel-ai/benchmark
// run-all` produces from a clean clone.
//
// Required env: at least one of OPENAI_API_KEY (for gpt-*) or
// ANTHROPIC_API_KEY (for claude-* + the default LLM judge). Local models via
// Ollama need no key — the `ollama:` prefix routes through the local server's
// OpenAI-compatible endpoint (http://localhost:11434/v1 by default). Examples:
//   --models ollama:qwen3.5,ollama:gemma4
//   --judge-model ollama:qwen3.5         (cost-free judge)
//
// User-hosted models (e.g. vLLM/TGI/LM Studio on EC2, or an AWS API-Gateway-fronted
// model) that expose an OpenAI-compatible endpoint are addressed by embedding the
// URL in the model string as `<baseURL>#<model-name>`. Optional bearer token via
// --model-api-key or AWS_BEDROCK_BEARER env; endpoints are auto-warmed before the run.
// Examples:
//   --models 'https://my-host:8000/v1#meta-llama/Llama-3.1-70B-Instruct'
//   --models 'https://models.example.com/v1#llama-3.1-70b' --model-api-key $TOKEN

import { existsSync } from "node:fs";
import { anthropic } from "@ai-sdk/anthropic";
import { createOpenAI, openai } from "@ai-sdk/openai";
import type { LanguageModel } from "ai";
import { config as loadEnv } from "dotenv";
import { type ParsedArgs, parseArgs, parseRejudgeArgs, resolveRunTarget } from "./cli-args.js";
import type { JudgePromptVariant } from "./judges/llm.js";
import {
  breakerLine,
  breakerThresholdFromEnv,
  retrySettingsFromEnv,
  retrySettingsLine,
} from "./llm-retry.js";
import { type CustomEndpoint, parseCustomEndpoint, warmUpModels } from "./model-endpoint.js";
import { buildRunnerModels, capsLine, loadModelCatalog } from "./output-limits.js";
import { resolveRepoPath } from "./paths.js";
import { loadModelPricing } from "./pricing.js";
import { rejudge } from "./rejudge.js";
import { doneLines, rerunSettingsLine, runExitCode } from "./rerun.js";
import { loadAgentRegistry, type RunnerConfig, run } from "./runner.js";
import type { ResolvedModel } from "./types.js";

loadEnv();

const OLLAMA_PREFIX = "ollama:";

interface ResolveOpts {
  ollamaBaseURL: string;
  /** Bearer token for user-hosted `<url>#<model>` endpoints (optional). */
  modelApiKey?: string;
}

/**
 * Resolve an Ollama model id (e.g. `ollama:qwen3.5`) into a Vercel AI SDK
 * `LanguageModel` that talks to the local Ollama server via its OpenAI-
 * compatible endpoint. The model id stored on the cell row keeps the
 * `ollama:` prefix so reports clearly distinguish local vs cloud models.
 *
 * Tool calling depends on the underlying model's native function-calling
 * support — Qwen / Llama families work well; Gemma is hit-or-miss. If a
 * local-model cell consistently logs zero tool calls, the model likely
 * isn't function-calling and the run is mainly measuring "did the model
 * write a coherent answer." That's still informative — just call it out
 * when reading the report.
 */
function resolveOllama(modelTag: string, baseURL: string): ResolvedModel {
  // `.chat(...)` forces the legacy `/v1/chat/completions` wire format. The
  // default factory call uses OpenAI's newer Responses API (typed items like
  // `item_reference`), which Ollama's OpenAI-compat endpoint doesn't speak.
  const provider = createOpenAI({ baseURL, apiKey: "ollama" });
  return { id: `${OLLAMA_PREFIX}${modelTag}`, model: provider.chat(modelTag) };
}

/**
 * Resolve a user-hosted model addressed as `<baseURL>#<model-name>` (e.g. a
 * vLLM/TGI/LM Studio server on EC2). Reuses the OpenAI-compatible SDK client
 * pointed at the caller's URL, exactly like {@link resolveOllama}, with
 * `.chat(...)` to force the legacy `/v1/chat/completions` wire format that
 * self-hosted servers implement (they rarely speak OpenAI's Responses API).
 *
 * The full `<url>#<model>` string is kept as the id so report rows are
 * unambiguous. Auth is optional: a bearer token from --model-api-key /
 * AWS_BEDROCK_BEARER when set, else a dummy key (unauthenticated endpoints).
 */
function resolveCustomEndpoint(raw: string, ep: CustomEndpoint, opts: ResolveOpts): ResolvedModel {
  const provider = createOpenAI({ baseURL: ep.baseURL, apiKey: opts.modelApiKey ?? "none" });
  return { id: raw, model: provider.chat(ep.modelName) };
}

function resolveModel(modelId: string, opts: ResolveOpts): ResolvedModel {
  const ep = parseCustomEndpoint(modelId);
  if (ep) {
    return resolveCustomEndpoint(modelId, ep, opts);
  }
  if (modelId.startsWith(OLLAMA_PREFIX)) {
    return resolveOllama(modelId.slice(OLLAMA_PREFIX.length), opts.ollamaBaseURL);
  }
  if (modelId.startsWith("claude")) {
    if (!process.env.ANTHROPIC_API_KEY) {
      throw new Error(`model ${modelId} requires ANTHROPIC_API_KEY (set in .env or shell)`);
    }
    return { id: modelId, model: anthropic(modelId) };
  }
  if (modelId.startsWith("gpt")) {
    if (!process.env.OPENAI_API_KEY) {
      throw new Error(`model ${modelId} requires OPENAI_API_KEY`);
    }
    return { id: modelId, model: openai(modelId) };
  }
  throw new Error(
    `unknown model provider for: ${modelId} ` +
      `(expected gpt-*, claude-*, ${OLLAMA_PREFIX}<tag>, or a user-hosted ` +
      `<baseURL>#<model-name> URL)`,
  );
}

/**
 * Pick the LLM judge model. `--no-judge` always wins. With `--judge-model X`
 * the user picks any provider (including `ollama:*`); without it the default
 * is Sonnet when ANTHROPIC_API_KEY is set, else no LLM judge (programmatic
 * judge still runs).
 */
function resolveJudge(parsed: ParsedArgs): LanguageModel | undefined {
  if (parsed.noJudge) return undefined;
  if (parsed.judgeModelId) {
    return resolveModel(parsed.judgeModelId, {
      ollamaBaseURL: parsed.ollamaBaseURL,
      modelApiKey: parsed.modelApiKey,
    }).model;
  }
  if (process.env.ANTHROPIC_API_KEY) return anthropic("claude-sonnet-4-6");
  return undefined;
}

/**
 * `--ephemeral` writes to a fresh per-run file under
 * `agent/results/ephemeral/<UTC-timestamp>.jsonl` instead of the
 * shared `agent.jsonl`. Designed for smoke tests / one-off campaigns where
 * the developer doesn't want to clobber the canonical output and shouldn't
 * have to think about `--force`. Conflicts with an explicit `--output`.
 */
function ephemeralOutputPath(): string {
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  return `agent/results/ephemeral/agent-${stamp}.jsonl`;
}

/** Default output path: `<input>.rejudged-<variant>.jsonl`, alongside the source. */
function defaultRejudgeOutput(input: string, variant: JudgePromptVariant): string {
  const dotJsonl = input.endsWith(".jsonl") ? input.slice(0, -".jsonl".length) : input;
  return `${dotJsonl}.rejudged-${variant}.jsonl`;
}

async function rejudgeMain(argv: string[]): Promise<void> {
  const parsed = parseRejudgeArgs(argv);
  // `--no-judge`: AST-only re-score — no LLM model resolved or called.
  const judgeModelId = parsed.noJudge ? undefined : (parsed.judgeModelId ?? "claude-sonnet-4-6");
  const judgeModel = judgeModelId
    ? resolveModel(judgeModelId, {
        ollamaBaseURL: parsed.ollamaBaseURL,
        modelApiKey: parsed.modelApiKey,
      }).model
    : undefined;

  const inputPath = resolveRepoPath(parsed.input);
  const outputPath = resolveRepoPath(
    parsed.out ?? defaultRejudgeOutput(parsed.input, parsed.promptVariant),
  );
  const corpusPath = resolveRepoPath(parsed.corpus);

  console.log(
    judgeModelId
      ? `rejudging ${inputPath} with ${judgeModelId} (${parsed.promptVariant}) + AST → ${outputPath}`
      : `re-scoring AST only (no LLM judge): ${inputPath} → ${outputPath}`,
  );
  const summary = await rejudge({
    inputPath,
    outputPath,
    corpusPath,
    judgeModel,
    promptVariant: parsed.promptVariant,
    judgeMaxOutputTokens: parsed.judgeMaxOutputTokens,
  });
  console.log(
    `done: ${summary.total} rows (${summary.ast_scored} AST-scored, ${summary.rejudged} LLM-rejudged, ` +
      `${summary.skipped_pass} kept as programmatic-pass, ${summary.skipped_error} errored left unjudged).`,
  );
}

async function runMain(): Promise<void> {
  const registry = await loadAgentRegistry();
  const knownArms = [...registry.keys()];
  const parsed = parseArgs(process.argv.slice(2), knownArms);
  // Output path, control-cache sources and tiers (reuse is ON by default),
  // --ratel-version and the judge cap.
  const target = resolveRunTarget(parsed, {
    resolve: resolveRepoPath,
    exists: existsSync,
    ephemeralOutput: ephemeralOutputPath,
  });
  const resolveOpts: ResolveOpts = {
    ollamaBaseURL: parsed.ollamaBaseURL,
    modelApiKey: parsed.modelApiKey,
  };
  // Caps are attached after resolution, so no resolver branch (e.g. an injected
  // early return) can skip them. Every model's cap is logged — none is implied.
  const models = buildRunnerModels(parsed.models, (m) => resolveModel(m, resolveOpts), {
    catalog: loadModelCatalog(),
    override: parsed.maxOutputTokens,
  });
  console.log(capsLine(models, parsed.maxOutputTokens));
  // Retry knobs (RATEL_LLM_RETRY_*, RATEL_CELL_TIMEOUT_GRACE_MS), validated and echoed once.
  const retry = retrySettingsFromEnv(process.env);
  console.log(retrySettingsLine(retry, parsed.timeoutMs));
  // Breaker (RATEL_ABORT_AFTER_CONSECUTIVE_ERRORS) and rerun flags, echoed once.
  const abortAfterConsecutiveErrors = breakerThresholdFromEnv(process.env);
  console.log(breakerLine(abortAfterConsecutiveErrors));
  console.log(rerunSettingsLine(parsed.rerun));
  // Warm any user-hosted endpoints once so early cells don't burn their timeout on
  // a cold start (no-op for cloud/ollama model ids).
  await warmUpModels(parsed.models, parsed.modelApiKey);
  const judgeModel = resolveJudge(parsed);

  if (!parsed.noJudge && !judgeModel) {
    console.warn(
      "warn: no LLM judge configured (set ANTHROPIC_API_KEY or pass --judge-model); " +
        "programmatic judge still active.",
    );
  }

  const cfg: RunnerConfig = {
    ...target,
    corpusPath: resolveRepoPath(parsed.corpus),
    scenarioLimit: parsed.scenarios,
    arms: parsed.arms,
    models,
    runsPerCell: parsed.runs,
    topK: parsed.topK,
    retriever: parsed.retriever,
    poolSizes: parsed.poolSizes,
    maxSteps: parsed.maxSteps,
    perRunTimeoutMs: parsed.timeoutMs,
    retry,
    rerun: parsed.rerun,
    abortAfterConsecutiveErrors,
    dollarGlobalCap: parsed.dollarGlobal,
    // Per-model rates from models.json (backend-aware). Empty when unpriced →
    // $0 rows, and the dollar cap simply doesn't bound the run.
    pricing: loadModelPricing(),
    force: parsed.force,
    judgeModel,
    noAst: parsed.noAst,
    seed: parsed.seed,
    concurrency: parsed.concurrency,
    logLevel: parsed.logLevel,
    registry,
  };

  console.log(
    `running ${parsed.arms.length} arms × ${models.length} models × ${parsed.runs} runs ` +
      `× ${parsed.poolSizes.length} pool size(s) [${parsed.poolSizes.join(",")}] ` +
      `over ≤ ${parsed.scenarios ?? "all"} scenarios at concurrency=${parsed.concurrency} ` +
      `→ ${target.outputPath}`,
  );
  if (parsed.ratelVersion !== undefined) {
    console.log(`ratel-version: control rows stamped as ${parsed.ratelVersion}`);
  }
  if (judgeModel && parsed.judgeMaxOutputTokens !== undefined) {
    console.log(`caps: judge=${parsed.judgeMaxOutputTokens}`);
  }
  const summary = await run(cfg);
  for (const line of doneLines(summary)) console.log(line);
  // A model the breaker aborted (gated, daily cap, outage) fails the run, after the summary.
  const exitCode = runExitCode(summary);
  if (exitCode !== 0) process.exitCode = exitCode;
}

async function main(): Promise<void> {
  const subcommand = process.argv[2];
  if (subcommand === "rejudge") {
    await rejudgeMain(process.argv.slice(3));
    return;
  }
  await runMain();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
