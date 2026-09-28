import { canonicalModelId } from "./model-identity.js";
import { normalizeSuiteRequest, resolveStablePair } from "./suite-contract.js";

export type WorkflowDispatchInputs = {
  runOnlyModels: string;
  excludeModels: string;
  notifyToEmails: string;
  modelConcurrency: string;
  campaignBudgetUsd: string;
  runId: string;
};

type JsonObject = Record<string, unknown>;

type PrepareWorkflowDispatchArgs = {
  rawInputs: WorkflowDispatchInputs;
  template: unknown;
  releases: unknown;
  benchmarkSha: string;
  generatedRunId: string;
};

const GIT_SHA = /^[0-9a-f]{40}$/;
const RUN_ID = /^[A-Za-z0-9][A-Za-z0-9_-]{0,99}$/;

export function prepareWorkflowDispatch({
  rawInputs,
  template,
  releases,
  benchmarkSha,
  generatedRunId,
}: PrepareWorkflowDispatchArgs) {
  if (!GIT_SHA.test(benchmarkSha)) throw new Error("benchmark SHA must be a full commit SHA");
  const base = jsonObject(template, "campaign template");
  if (base.schemaVersion !== 1) throw new Error("campaign template must use schemaVersion 1");
  if (typeof base.awsSha !== "string" || !GIT_SHA.test(base.awsSha)) {
    throw new Error("campaign template AWS SHA must be a full commit SHA");
  }
  const runOnlyModels = csv(rawInputs.runOnlyModels)?.map(canonicalModelId);
  const excludeModels = csv(rawInputs.excludeModels)?.map(canonicalModelId) ?? [];
  const notifyToEmails = csv(rawInputs.notifyToEmails) ?? [];
  if (notifyToEmails.length > 20) {
    throw new Error("notifyToEmails must contain at most 20 email addresses");
  }
  const normalized = normalizeSuiteRequest({
    schemaVersion: 1,
    ...(runOnlyModels === undefined ? {} : { runOnlyModels }),
    excludeModels,
    notifyToEmails,
    ...(rawInputs.modelConcurrency === ""
      ? {}
      : { modelConcurrency: numberInput(rawInputs.modelConcurrency, "modelConcurrency") }),
    ...(rawInputs.campaignBudgetUsd === ""
      ? {}
      : { campaignBudgetUsd: numberInput(rawInputs.campaignBudgetUsd, "campaignBudgetUsd") }),
  });
  const runId = rawInputs.runId.trim() || generatedRunId;
  if (!RUN_ID.test(runId)) throw new Error("runId must contain only letters, numbers, _ or -");
  const pair = resolveStablePair(releases);
  const options = {
    runId,
    ...(runOnlyModels === undefined ? {} : { runOnlyModels }),
    excludeModels,
    notifyToEmails: normalized.notifyToEmails,
    modelConcurrency: normalized.modelConcurrency,
    campaignBudgetUsd: normalized.campaignBudgetUsd,
  };
  return {
    awsSha: base.awsSha,
    selectedModels: normalized.models,
    options,
    template: {
      ...base,
      benchmarkSha,
      ratel: {
        sdkVersion: pair.sdk.version,
        sdkIntegrity: pair.sdk.integrity,
        coreVersion: pair.core.version,
        coreChecksum: pair.core.checksum,
      },
    },
  };
}

export function summarizeCampaignLaunch(
  output: string,
  repository: string,
  workflowRunId: string,
): string {
  const lines = output.trim().split(/\r?\n/);
  const [runId, executionLine, consoleLine, resultUri] = lines;
  if (
    !RUN_ID.test(runId ?? "") ||
    !executionLine?.startsWith("execution: arn:aws:states:") ||
    !consoleLine?.startsWith("console: https://") ||
    !resultUri?.startsWith("s3://")
  ) {
    throw new Error("launcher returned an invalid response");
  }
  const executionUrl = consoleLine.slice("console: ".length);
  const arn = executionLine.slice("execution: ".length);
  const region = arn.split(":")[3];
  const { bucket, key } = parseS3Uri(resultUri);
  const resultUrl = `https://s3.console.aws.amazon.com/s3/object/${bucket}?region=${encodeURIComponent(region)}&prefix=${encodeURIComponent(key)}`;
  const logsUrl = `https://github.com/${repository}/actions/runs/${workflowRunId}`;
  return [
    "## Benchmark campaign launched",
    "",
    `Run ID: \`${runId}\``,
    "",
    `- [Workflow logs](${logsUrl})`,
    `- [Step Functions execution](${executionUrl})`,
    `- [Result object](${resultUrl}) (available after completion)`,
    "",
    "Submission returned after StartExecution; benchmark workers continue asynchronously.",
  ].join("\n");
}

function csv(value: string): string[] | undefined {
  if (value.trim() === "") return undefined;
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function numberInput(value: string, name: string): number {
  if (!/^(?:0|[1-9]\d*)(?:\.\d+)?$/.test(value)) throw new Error(`${name} must be numeric`);
  return Number(value);
}

function jsonObject(value: unknown, name: string): JsonObject {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${name} must be a JSON object`);
  }
  return value as JsonObject;
}

function parseS3Uri(uri: string): { bucket: string; key: string } {
  const match = /^s3:\/\/([a-z0-9][a-z0-9.-]{2,62})\/(.+)$/.exec(uri);
  if (!match) throw new Error("launcher returned an invalid result URI");
  return { bucket: match[1], key: match[2] };
}
