import { describe, expect, it } from "vitest";
import { cellKeyOf, controlKeyOf, modelRouteOfRow } from "./cell-key.js";
import type { CellResult } from "./types.js";

const row = (model: string, provider?: string): CellResult =>
  ({
    ratel_version: "0.12.0",
    scenario_id: "s1",
    arm: "control-baseline",
    model,
    provider,
    run_index: 0,
    pool_size: 30,
  }) as CellResult;

describe("provider-qualified cell identity", () => {
  it("keeps the same Claude name on three serving routes separate", () => {
    const routes = [
      row("bedrock/claude-sonnet-5", "amazon-bedrock"),
      row("anthropic/claude-sonnet-5", "anthropic.messages"),
      row("gcp/claude-sonnet-5", "vertex.anthropic.messages"),
    ];
    expect(new Set(routes.map(cellKeyOf)).size).toBe(3);
    expect(new Set(routes.map(controlKeyOf)).size).toBe(3);
  });

  it("migrates bare historical names only with recorded serving evidence", () => {
    expect(modelRouteOfRow(row("claude-sonnet-5", "anthropic.messages"))).toBe(
      "anthropic/claude-sonnet-5",
    );
    expect(modelRouteOfRow(row("claude-sonnet-5", "vertex.anthropic.messages"))).toBe(
      "gcp/claude-sonnet-5",
    );
    expect(modelRouteOfRow(row("claude-sonnet-5", "amazon-bedrock"))).toBe(
      "bedrock/claude-sonnet-5",
    );
    expect(modelRouteOfRow(row("claude-sonnet-5"))).toBe("claude-sonnet-5");
    expect(modelRouteOfRow(row("gpt-6-sol", "openai.responses"))).toBe("gpt-6-sol");
    expect(
      modelRouteOfRow({
        model: "claude-sonnet-5",
        provider: "anthropic.messages",
        serving_provider: "bedrock",
      }),
    ).toBe("claude-sonnet-5");
    expect(cellKeyOf(row("claude-sonnet-5"))).not.toBe(cellKeyOf(row("bedrock/claude-sonnet-5")));
  });
});
