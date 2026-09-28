import { appendFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { REPO_ROOT } from "./paths.js";
import {
  prepareWorkflowDispatch,
  summarizeCampaignLaunch,
  type WorkflowDispatchInputs,
} from "./workflow-dispatch.js";

const CAMPAIGN_DIR = resolve(REPO_ROOT, ".campaign");
const RELEASES_FILE = resolve(REPO_ROOT, ".github/benchmark-release-metadata.json");

export function runWorkflowDispatchCli(action: string, target?: string): void {
  if (action === "prepare" && target === undefined) {
    prepare();
    return;
  }
  if (action === "summarize" && target) {
    summarize(target);
    return;
  }
  throw new Error("usage: workflow-dispatch-cli prepare | summarize <launch-output>");
}

function prepare(): void {
  const templateJson = requiredEnv("CAMPAIGN_TEMPLATE_JSON");
  const prepared = prepareWorkflowDispatch({
    rawInputs: workflowInputs(process.env),
    template: parseJson(templateJson, "CAMPAIGN_TEMPLATE_JSON"),
    releases: parseJson(readFileSync(RELEASES_FILE, "utf8"), "release metadata"),
    benchmarkSha: requiredEnv("BENCHMARK_SHA"),
    generatedRunId: `gh-${requiredEnv("GITHUB_RUN_ID")}-${requiredEnv("GITHUB_RUN_ATTEMPT")}`,
  });
  mkdirSync(CAMPAIGN_DIR, { recursive: true, mode: 0o700 });
  writePrivateJson("request-template.json", prepared.template);
  writePrivateJson("options.json", prepared.options);
  githubOutput("aws_sha", prepared.awsSha);
  githubOutput("run_id", prepared.options.runId);
}

function summarize(target: string): void {
  const markdown = summarizeCampaignLaunch(
    readFileSync(resolve(REPO_ROOT, target), "utf8"),
    requiredEnv("GITHUB_REPOSITORY"),
    requiredEnv("GITHUB_RUN_ID"),
  );
  appendFileSync(requiredEnv("GITHUB_STEP_SUMMARY"), `${markdown}\n`, "utf8");
  process.stdout.write(`${markdown}\n`);
}

function workflowInputs(env: NodeJS.ProcessEnv): WorkflowDispatchInputs {
  return {
    runOnlyModels: env.RUN_ONLY_MODELS ?? "",
    excludeModels: env.EXCLUDE_MODELS ?? "",
    notifyToEmails: env.NOTIFY_TO_EMAILS ?? "",
    modelConcurrency: env.MODEL_CONCURRENCY ?? "",
    campaignBudgetUsd: env.CAMPAIGN_BUDGET_USD ?? "",
    runId: env.CAMPAIGN_RUN_ID ?? "",
  };
}

function writePrivateJson(name: string, value: unknown): void {
  writeFileSync(resolve(CAMPAIGN_DIR, name), `${JSON.stringify(value, null, 2)}\n`, {
    encoding: "utf8",
    mode: 0o600,
  });
}

function parseJson(value: string, name: string): unknown {
  try {
    return JSON.parse(value);
  } catch (error) {
    throw new Error(`${name} is invalid JSON: ${error instanceof Error ? error.message : error}`);
  }
}

function requiredEnv(name: string): string {
  const value = process.env[name];
  if (!value) throw new Error(`${name} is required`);
  return value;
}

function githubOutput(name: string, value: string): void {
  appendFileSync(requiredEnv("GITHUB_OUTPUT"), `${name}=${value}\n`, "utf8");
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    runWorkflowDispatchCli(process.argv[2] ?? "", process.argv[3]);
  } catch (error) {
    process.stderr.write(
      `workflow-dispatch: ERROR: ${error instanceof Error ? error.message : error}\n`,
    );
    process.exitCode = 1;
  }
}
