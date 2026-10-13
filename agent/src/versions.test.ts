import { describe, expect, it } from "vitest";
import { resolveCoreVersions } from "./versions.js";

describe("core version provenance", () => {
  const lock = `[[package]]\nname = "ratel-ai-core"\nversion = "0.11.0"\n`;

  it("keeps the resolved crate release separate from a report label", () => {
    expect(resolveCoreVersions(lock, "0.11.0-hybrid")).toEqual({
      label: "0.11.0-hybrid",
      resolved: "0.11.0",
    });
  });

  it("uses the resolved release as the default label", () => {
    expect(resolveCoreVersions(lock)).toEqual({ label: "0.11.0", resolved: "0.11.0" });
  });
});
