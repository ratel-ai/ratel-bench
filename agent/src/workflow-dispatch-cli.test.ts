import { writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { afterEach, expect, it, vi } from "vitest";
import { REPO_ROOT } from "./paths.js";
import { runWorkflowDispatchCli } from "./workflow-dispatch-cli.js";

vi.mock("node:fs", async (importOriginal) => ({
  ...(await importOriginal<typeof import("node:fs")>()),
  appendFileSync: vi.fn(),
  mkdirSync: vi.fn(),
  writeFileSync: vi.fn(),
}));

afterEach(() => {
  vi.unstubAllEnvs();
  vi.clearAllMocks();
});

it.each([
  [undefined, 16],
  ["", 16],
  ["5", 5],
])("writes LLM_CONCURRENCY %j to options.json as %i", (value, expected) => {
  prepareEnv(value);

  runWorkflowDispatchCli("prepare");

  const optionsJson = vi
    .mocked(writeFileSync)
    .mock.calls.find(([file]) => file === resolve(REPO_ROOT, ".campaign/options.json"))?.[1];
  expect(JSON.parse(String(optionsJson))).toMatchObject({ llmConcurrency: expected });
});

it("rejects malformed LLM_CONCURRENCY before writing dispatch files", () => {
  prepareEnv("8; echo bad");

  expect(() => runWorkflowDispatchCli("prepare")).toThrow(/llmConcurrency/);
  expect(writeFileSync).not.toHaveBeenCalled();
});

function prepareEnv(llmConcurrency: string | undefined): void {
  vi.stubEnv(
    "CAMPAIGN_TEMPLATE_JSON",
    JSON.stringify({ schemaVersion: 1, awsSha: "b".repeat(40) }),
  );
  vi.stubEnv("BENCHMARK_SHA", "a".repeat(40));
  vi.stubEnv("GITHUB_RUN_ID", "123");
  vi.stubEnv("GITHUB_RUN_ATTEMPT", "1");
  vi.stubEnv("GITHUB_OUTPUT", "github-output");
  for (const name of [
    "RUN_ONLY_MODELS",
    "EXCLUDE_MODELS",
    "NOTIFY_TO_EMAILS",
    "MODEL_CONCURRENCY",
    "CAMPAIGN_BUDGET_USD",
    "CAMPAIGN_RUN_ID",
  ]) {
    vi.stubEnv(name, "");
  }
  vi.stubEnv("LLM_CONCURRENCY", llmConcurrency);
}
