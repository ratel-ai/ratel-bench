import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative, resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { REPO_ROOT } from "./paths.js";
import { prepareWorkflowDispatch, summarizeCampaignLaunch } from "./workflow-dispatch.js";

// Repo has no CI: local/Loop-run tripwire, not an IAM boundary.
const PRIVILEGED_WORKFLOW_MARKERS = [
  /id-token:\s*['"]?write/,
  /permissions:\s*['"]?write-all/,
  /role-to-assume/,
  /states:StartExecution/,
  /stepfunctions\s+start-execution/,
  /bench-suite\.sh/,
];
const SHA_A = "a".repeat(40);
const SHA_B = "b".repeat(40);
const template = {
  schemaVersion: 1,
  runId: "replaced-at-dispatch",
  benchmarkSha: "0".repeat(40),
  awsSha: SHA_B,
  ratel: {
    sdkVersion: "0.0.0",
    sdkIntegrity: "old",
    coreVersion: "0.0.0",
    coreChecksum: "old",
  },
  harness: { frozen: true },
};
const releases = {
  sdk: [
    { version: "0.12.0", integrity: "sha512-stable" },
    { version: "0.13.0-rc.7", integrity: "sha512-rc" },
  ],
  core: [{ version: "0.11.0", checksum: "checksum-stable" }],
  compatible: [
    { sdk: "0.13.0-rc.7", core: "0.11.0" },
    { sdk: "0.12.0", core: "0.11.0" },
  ],
};

describe("prepareWorkflowDispatch", () => {
  it("matches CLI selection defaults and freezes exact sources and latest stable pair", () => {
    const prepared = prepareWorkflowDispatch({
      rawInputs: {
        runOnlyModels: "",
        excludeModels: "",
        notifyToEmails: "",
        modelConcurrency: "",
        campaignBudgetUsd: "",
        runId: "",
      },
      template,
      releases,
      benchmarkSha: SHA_A,
      generatedRunId: "gh-123-1",
    });

    expect(prepared.options).toEqual({
      runId: "gh-123-1",
      excludeModels: [],
      notifyToEmails: ["dev@ratel.sh"],
      modelConcurrency: 1,
      campaignBudgetUsd: 1000,
    });
    expect(prepared.template).toMatchObject({
      benchmarkSha: SHA_A,
      awsSha: SHA_B,
      harness: { frozen: true },
      ratel: {
        sdkVersion: "0.12.0",
        sdkIntegrity: "sha512-stable",
        coreVersion: "0.11.0",
        coreChecksum: "checksum-stable",
      },
    });
    expect(prepared.awsSha).toBe(SHA_B);
  });

  it("serializes hostile values as data while exclusion wins and recipients deduplicate", () => {
    const prepared = prepareWorkflowDispatch({
      rawInputs: {
        runOnlyModels: "bedrock/openai.gpt-6-sol,xai/model;touch-nope",
        excludeModels: "bedrock/openai.gpt-6-sol",
        notifyToEmails: " Ops@example.com,ops@example.com ",
        modelConcurrency: "2",
        campaignBudgetUsd: "125.5",
        runId: "manual_run-1",
      },
      template,
      releases,
      benchmarkSha: SHA_A,
      generatedRunId: "unused",
    });

    expect(prepared.options).toEqual({
      runId: "manual_run-1",
      runOnlyModels: ["bedrock/openai.gpt-6-sol", "xai/model;touch-nope"],
      excludeModels: ["bedrock/openai.gpt-6-sol"],
      notifyToEmails: ["dev@ratel.sh", "ops@example.com"],
      modelConcurrency: 2,
      campaignBudgetUsd: 125.5,
    });
    expect(prepared.selectedModels).toEqual(["xai/model;touch-nope"]);
    expect(JSON.parse(JSON.stringify(prepared.options)).runOnlyModels).toContain(
      "xai/model;touch-nope",
    );
  });

  it("rejects malformed source pins, numbers, emails and empty final selection", () => {
    const base = {
      rawInputs: {
        runOnlyModels: "bedrock/a",
        excludeModels: "bedrock/a",
        notifyToEmails: "",
        modelConcurrency: "1",
        campaignBudgetUsd: "1000",
        runId: "run-1",
      },
      template,
      releases,
      benchmarkSha: SHA_A,
      generatedRunId: "unused",
    };
    expect(() => prepareWorkflowDispatch(base)).toThrow(/selection is empty/);
    expect(() => prepareWorkflowDispatch({ ...base, benchmarkSha: "main" })).toThrow(
      /benchmark SHA/,
    );
    expect(() =>
      prepareWorkflowDispatch({
        ...base,
        rawInputs: { ...base.rawInputs, excludeModels: "", modelConcurrency: "1; echo bad" },
      }),
    ).toThrow(/modelConcurrency/);
    expect(() =>
      prepareWorkflowDispatch({
        ...base,
        rawInputs: {
          ...base.rawInputs,
          excludeModels: "",
          notifyToEmails: Array.from({ length: 21 }, (_, index) => `a${index}@example.com`).join(
            ",",
          ),
        },
      }),
    ).toThrow(/at most 20/);
  });
});

it("renders immediate workflow, execution and immutable result links", () => {
  const summary = summarizeCampaignLaunch(
    [
      "run-123",
      "execution: arn:aws:states:eu-central-1:123:execution:campaign:run-123",
      "console: https://eu-central-1.console.aws.amazon.com/states/example",
      "s3://private-results/runs/run-123/result.json",
    ].join("\n"),
    "ratel-ai/ratel-bench-aws",
    "456",
  );
  expect(summary).toContain("Run ID: `run-123`");
  expect(summary).toContain(
    "[Workflow logs](https://github.com/ratel-ai/ratel-bench-aws/actions/runs/456)",
  );
  expect(summary).toContain(
    "[Step Functions execution](https://eu-central-1.console.aws.amazon.com/states/example)",
  );
  expect(summary).toContain(
    "[Result object](https://s3.console.aws.amazon.com/s3/object/private-results?region=eu-central-1&prefix=runs%2Frun-123%2Fresult.json)",
  );
});

it("benchmark repository holds no privileged campaign launch workflow", () => {
  const workflowsDir = resolve(REPO_ROOT, ".github/workflows");
  expect(existsSync(join(workflowsDir, "run-benchmark.yml"))).toBe(false);
  if (!existsSync(workflowsDir)) return;
  const offenders = readdirSync(workflowsDir, { recursive: true, encoding: "utf8" })
    .map((entry) => join(workflowsDir, entry))
    .filter((path) => statSync(path).isFile())
    .filter((path) => isPrivilegedWorkflow(readFileSync(path, "utf8")))
    .map((path) => relative(REPO_ROOT, path));
  expect(offenders).toEqual([]);
});

describe("isPrivilegedWorkflow", () => {
  it.each([
    ["quoted id-token", 'permissions:\n  id-token: "write"'],
    ["padded id-token", "permissions:\n  id-token:   write"],
    ["unspaced id-token", "permissions:\n  id-token:write"],
    ["write-all", "permissions: write-all"],
    ["role assumption", "with:\n  role-to-assume: arn:aws:iam::1:role/x"],
    ["sfn IAM action", "Action: states:StartExecution"],
    ["sfn CLI", "run: aws stepfunctions start-execution --state-machine-arn x"],
    ["bench-suite", "run: ./scripts/bench-suite.sh"],
  ])("flags %s", (_, content) => {
    expect(isPrivilegedWorkflow(content)).toBe(true);
  });

  it("allows benign read-only workflow", () => {
    expect(isPrivilegedWorkflow("permissions:\n  contents: read\nsteps:\n  - run: pnpm test")).toBe(
      false,
    );
  });
});

function isPrivilegedWorkflow(content: string): boolean {
  return PRIVILEGED_WORKFLOW_MARKERS.some((marker) => marker.test(content));
}
