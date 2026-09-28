import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
  DEFAULT_BFCL_MODELS,
  DEFAULT_JUDGE_MODEL,
  DEFAULT_SRAGENTS_MODELS,
  runAllProviderModels,
} from "./model-defaults.js";
import { REPO_ROOT } from "./paths.js";

describe("internally generated model routes", () => {
  it("keeps campaign and judge defaults on Bedrock", () => {
    const selected = JSON.parse(readFileSync(resolve(REPO_ROOT, "models.json"), "utf8")).run;
    expect(DEFAULT_BFCL_MODELS).toEqual(selected.map(({ id }: { id: string }) => id));
    expect(DEFAULT_SRAGENTS_MODELS).toEqual(DEFAULT_BFCL_MODELS);
    expect(DEFAULT_JUDGE_MODEL).toBe("bedrock/anthropic.claude-sonnet-5");
  });

  it("preserves the old run-all direct routes when their keys are configured", () => {
    expect(runAllProviderModels({ anthropic: true, openai: true })).toEqual([
      "anthropic/claude-sonnet-4-6",
      "openai/gpt-5.4-mini",
    ]);
    expect(runAllProviderModels({ anthropic: false, openai: true })).toEqual([
      "openai/gpt-5.4-mini",
    ]);
  });
});
