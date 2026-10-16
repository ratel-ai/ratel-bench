import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { embeddingArtifactOption, loadEmbeddingArtifact } from "./embedding-artifact.js";
import { REPO_ROOT } from "./paths.js";
import { resetForTests, selectVersion } from "./sdk/resolve.js";

const dirs: string[] = [];

beforeEach(() => resetForTests());
afterEach(() => {
  resetForTests();
  vi.restoreAllMocks();
  for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
});

describe("embeddingArtifactOption", () => {
  it.each([
    "0.4.0",
    "0.5.0",
  ])("skips reading a missing artifact and warns once on legacy SDK %s", async (version) => {
    selectVersion(version);
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    for (let i = 0; i < 2; i++) {
      await expect(
        embeddingArtifactOption({ path: "/missing/legacy.rat1", method: "semantic" }),
      ).resolves.toBeUndefined();
    }
    expect(warn).toHaveBeenCalledExactlyOnceWith(
      `SDK ${version} lacks experimentalEmbeddingArtifact; embedding inline`,
    );
  });

  it("still fails on a missing artifact for a supporting SDK", async () => {
    selectVersion("0.12.0");
    await expect(
      embeddingArtifactOption({ path: "/missing/supported.rat1", method: "hybrid" }),
    ).rejects.toThrow(/ENOENT/);
  });

  it("ignores the artifact for BM25 without reading it", async () => {
    selectVersion("0.12.0");
    await expect(
      embeddingArtifactOption({ path: "/missing/bm25.rat1", method: "bm25" }),
    ).resolves.toBeUndefined();
  });

  it("rejects --embedding combined with --embedding-artifact up front", async () => {
    selectVersion("0.12.0");
    await expect(
      embeddingArtifactOption({
        path: "/missing/combo.rat1",
        method: "semantic",
        embedding: { huggingface: "Xenova/all-MiniLM-L6-v2" },
      }),
    ).rejects.toThrow(/--embedding cannot be combined with --embedding-artifact/);
  });
});

describe("loadEmbeddingArtifact", () => {
  it("retries a rejected read once the file appears", async () => {
    const dir = mkdtempSync(join(REPO_ROOT, "agent/.embedding-artifact-option-test-"));
    dirs.push(dir);
    const path = join(dir, "late.rat1");
    await expect(loadEmbeddingArtifact(path)).rejects.toThrow(/ENOENT/);
    writeFileSync(path, "late bytes");
    await expect(loadEmbeddingArtifact(path)).resolves.toEqual(Buffer.from("late bytes"));
  });
});
