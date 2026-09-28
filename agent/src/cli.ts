// CLI entry. Wires AI SDK provider models to the runner. Default scenario
// corpus is the ingested MetaTool snapshot, which `pnpm -F @ratel-ai/benchmark
// run-all` produces from a clean clone.
//
// Direct OpenAI and Anthropic routes require their respective API keys. Local models via
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
import { DEFAULT_JUDGE_MODEL } from "./model-defaults.js";
import { warmUpModels } from "./model-endpoint.js";
import { resolveModel } from "./model-factory.js";
import { buildRunnerModels, capsLine, loadModelCatalog } from "./output-limits.js";
import { resolveRepoPath } from "./paths.js";
import { loadModelPricing } from "./pricing.js";
import { rejudge } from "./rejudge.js";
import { doneLines, rerunSettingsLine, runExitCode } from "./rerun.js";
import { loadAgentRegistry, type RunnerConfig, run } from "./runner.js";
import { selectVersion, validateSdk } from "./sdk/resolve.js";

loadEnv();

interface ResolveOpts {
  ollamaBaseURL: string;
  /** Bearer token for user-hosted `<url>#<model>` endpoints (optional). */
  modelApiKey?: string;
}

/**
 * Pick the LLM judge model. `--no-judge` always wins. With `--judge-model X`
 * the user picks any provider (including `ollama:*`); without it the default
 * Bedrock judge is enabled only when the Bedrock backend is configured.
 */
function resolveJudge(parsed: ParsedArgs): LanguageModel | undefined {
  if (parsed.noJudge) return undefined;
  if (parsed.judgeModelId) {
    return resolveModel(parsed.judgeModelId, {
      ollamaBaseURL: parsed.ollamaBaseURL,
      modelApiKey: parsed.modelApiKey,
    }).model;
  }
  if (process.env.RATEL_LLM_BACKEND === "bedrock") {
    return resolveModel(DEFAULT_JUDGE_MODEL, {
      ollamaBaseURL: parsed.ollamaBaseURL,
      modelApiKey: parsed.modelApiKey,
    }).model;
  }
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
  const judgeModelId = parsed.noJudge ? undefined : (parsed.judgeModelId ?? DEFAULT_JUDGE_MODEL);
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
  selectVersion(parsed.sdkVersion ?? "");
  await validateSdk();
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
      "warn: no LLM judge configured (select a Bedrock backend or pass --judge-model); " +
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
