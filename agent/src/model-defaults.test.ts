import { describe, expect, it } from "vitest";
import {
  DEFAULT_BFCL_MODELS,
  DEFAULT_JUDGE_MODEL,
  DEFAULT_SRAGENTS_MODEL,
  runAllProviderModels,
} from "./model-defaults.js";

describe("internally generated model routes", () => {
  it("keeps campaign and judge defaults on Bedrock", () => {
    expect(DEFAULT_BFCL_MODELS).toEqual(["bedrock/claude-sonnet-5", "bedrock/claude-haiku-4-5"]);
    expect(DEFAULT_SRAGENTS_MODEL).toBe("bedrock/claude-sonnet-5");
    expect(DEFAULT_JUDGE_MODEL).toBe("bedrock/claude-sonnet-5");
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
