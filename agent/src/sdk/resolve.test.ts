import { readFileSync } from "node:fs";
import { afterEach, describe, expect, it } from "vitest";
import { meter } from "../metering.js";
import {
  assertSdkApi,
  loadSdk,
  resetForTests,
  type SdkModule,
  sdkVersion,
  selectVersion,
  validateSdk,
} from "./resolve.js";

afterEach(resetForTests);

describe("SDK version selection", () => {
  it("pins the verified npm artifact and keeps historical alias targets", () => {
    const manifest = JSON.parse(
      readFileSync(new URL("../../package.json", import.meta.url), "utf8"),
    );
    const lock = readFileSync(new URL("../../../pnpm-lock.yaml", import.meta.url), "utf8");
    expect(manifest.dependencies["@ratel-ai/sdk-0.12.0"]).toBe("npm:@ratel-ai/sdk@0.12.0");
    expect(manifest.dependencies["@ratel-ai/sdk-0.2.0"]).toBe("npm:@ratel-ai/sdk@0.2.0");
    expect(manifest.dependencies["@ratel-ai/sdk-0.4.0"]).toBe("npm:@ratel-ai/sdk@0.4.0");
    expect(manifest.dependencies["@ratel-ai/sdk-0.5.0"]).toBe("npm:@ratel-ai/sdk@0.5.0");
    const lockedRelease = lock.split("  '@ratel-ai/sdk@0.12.0':\n")[1]?.split("\n\n")[0];
    expect(lockedRelease).toContain(
      "sha512-7gapjM5oMpS5JEYstQMBltidT2eIDILHjLYQnkbvjVz9rdlRBh1uva5IyyBW/+6KEioiTBzgH5gIpQUsvkw03g==",
    );
  });
  it("loads the exact stable 0.12.0 release without changing historical aliases", async () => {
    selectVersion("0.12.0");
    expect(sdkVersion()).toBe("0.12.0");
    expect((await loadSdk()).ToolCatalog).toBeTypeOf("function");

    resetForTests();
    selectVersion("0.4.0");
    expect(sdkVersion()).toBe("0.4.0");
    resetForTests();
    expect(sdkVersion()).toBe("0.4.0");
  });

  it("rejects an uninstalled release before loading any SDK", () => {
    expect(() => selectVersion("99.99.99")).toThrow(/no alias.*installed/);
    expect(() => selectVersion("latest")).toThrow(/exact release/);
    expect(() => selectVersion("^0.12.0")).toThrow(/exact release/);
  });

  it("preflights the native catalog and gateway API before inference", async () => {
    selectVersion("0.12.0");
    await expect(validateSdk()).resolves.toBeUndefined();
    expect(() => assertSdkApi({ ToolCatalog: class {} } as SdkModule)).toThrow(
      /unsupported SDK API.*SkillCatalog/,
    );
    const selected = await loadSdk();
    expect(() =>
      assertSdkApi({ ...selected, SEARCH_TOOLS_ID: "changed" } as unknown as SdkModule),
    ).toThrow(/unsupported SDK API.*search_tools/);
  });

  it("stamps measured rows with the selected package version", async () => {
    selectVersion("0.12.0");
    const { cell } = await meter(
      {
        scenarioId: "fixture",
        arm: "control-baseline",
        model: "offline",
        runIndex: 0,
        catalogSize: 1,
        poolSize: 1,
        seed: 0,
      },
      async () => ({ steps: [] }),
    );
    expect(cell.ratel_version).toBe("0.12.0");
  });
});
