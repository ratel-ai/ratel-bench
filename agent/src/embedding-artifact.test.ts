import { execFile } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, unlinkSync, writeFileSync } from "node:fs";
import { join, relative } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { afterEach, describe, expect, it } from "vitest";
import { loadEmbeddingArtifact } from "./embedding-artifact.js";
import { REPO_ROOT } from "./paths.js";
import { loadSdk, resetForTests, selectVersion } from "./sdk/resolve.js";

const dirs: string[] = [];
const exec = promisify(execFile);

afterEach(() => {
  resetForTests();
  for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
});

describe("embedding-artifact CLI", () => {
  it.each([
    "0.4.0",
    "0.5.0",
  ])("reports SDK %s as unsupported before reading the universe", async (version) => {
    const result = await exec(
      process.execPath,
      [
        "--import",
        "tsx",
        "src/embedding-artifact-cli.ts",
        "build",
        "--benchmark",
        "bfcl",
        "--sdk-version",
        version,
        "--corpus",
        "/missing/bfcl.jsonl",
        "--output",
        "/unused/artifact.rat1",
      ],
      { cwd: new URL("..", import.meta.url) },
    ).catch((error) => error);
    expect(result.code).toBe(3);
    expect(result.stderr).toContain(`unsupported: SDK ${version}`);
    expect(result.stderr).toContain("experimentalBuildEmbeddingArtifact");
  });
});

describe("BFCL embedding artifact", () => {
  it("warms every tool from the full corpus using the candidate registration shape", async () => {
    const dir = mkdtempSync(
      fileURLToPath(new URL("../.embedding-artifact-test-", import.meta.url)),
    );
    dirs.push(dir);
    const corpus = join(dir, "bfcl.jsonl");
    const output = join(dir, "nested", "bfcl.rat1");
    const tools = [
      {
        id: "files.read",
        name: "read_file",
        description: "Read a file from disk",
        input_schema: { type: "object", properties: { path: { type: "string" } } },
        output_schema: { type: "object", properties: { contents: { type: "string" } } },
      },
      {
        id: "mail.send",
        name: "send_email",
        description: "Send an email message",
        input_schema: { type: "object", properties: { recipient: { type: "string" } } },
      },
    ];
    writeFileSync(
      corpus,
      tools
        .map((tool, i) =>
          JSON.stringify({
            id: `scenario-${i}`,
            prompt: tool.description,
            candidate_pool: [tool],
            gold_tools: [tool.id],
          }),
        )
        .join("\n"),
    );

    await exec(
      process.execPath,
      [
        "--import",
        "tsx",
        "src/embedding-artifact-cli.ts",
        "build",
        "--benchmark",
        "bfcl",
        "--sdk-version",
        "0.12.0",
        "--corpus",
        corpus,
        "--output",
        output,
      ],
      { cwd: new URL("..", import.meta.url) },
    );

    const bytes = readFileSync(output);
    expect(bytes.subarray(0, 4).toString()).toBe("RAT1");
    selectVersion("0.12.0");
    const sdk = (await loadSdk()) as unknown as typeof import("@ratel-ai/sdk-0.12.0");
    const catalog = new sdk.ToolCatalog({
      method: "semantic",
      experimentalEmbeddingArtifact: { bytes, onMiss: "error" },
    });
    await expect(
      catalog.register([
        {
          id: "files.read",
          name: "read_file",
          description: "Read a file from disk",
          inputSchema: { type: "object", properties: { path: { type: "string" } } },
          outputSchema: { type: "object", properties: { contents: { type: "string" } } },
          execute: async () => ({}),
        },
        {
          id: "mail.send",
          name: "send_email",
          description: "Send an email message",
          inputSchema: { type: "object", properties: { recipient: { type: "string" } } },
          outputSchema: {},
          execute: async () => ({}),
        },
      ]),
    ).resolves.toBeUndefined();
    expect((await catalog.searchAsync("read a file from disk", 1))[0]?.toolId).toBe("files.read");
  }, 60_000);
});

describe("SR-Agents embedding artifact", () => {
  it("warms the full skill universe with bodies and extra fields stripped like candidates", async () => {
    const dir = mkdtempSync(
      fileURLToPath(new URL("../.embedding-artifact-test-", import.meta.url)),
    );
    dirs.push(dir);
    const skills = join(dir, "skills.jsonl");
    const output = join(dir, "nested", "skills.rat1");
    writeFileSync(
      skills,
      [
        {
          id: "edit",
          name: "Edit file",
          description: "Edit a local file",
          body: "Body is not indexed by candidates",
          tags: ["ignored"],
        },
        {
          id: "travel",
          name: "Book flight",
          description: "Reserve air travel",
          body: "Different body",
        },
      ]
        .map((skill) => JSON.stringify(skill))
        .join("\n"),
    );

    await exec(
      process.execPath,
      [
        "--import",
        "tsx",
        "src/embedding-artifact-cli.ts",
        "build",
        "--benchmark",
        "sragents",
        "--sdk-version",
        "0.12.0",
        "--skills",
        skills,
        "--output",
        output,
      ],
      { cwd: new URL("..", import.meta.url) },
    );

    const bytes = readFileSync(output);
    expect(bytes.subarray(0, 4).toString()).toBe("RAT1");
    selectVersion("0.12.0");
    const sdk = (await loadSdk()) as unknown as typeof import("@ratel-ai/sdk-0.12.0");
    const catalog = new sdk.SkillCatalog({
      method: "semantic",
      experimentalEmbeddingArtifact: { bytes, onMiss: "error" },
    });
    await expect(
      catalog.register([
        { id: "edit", name: "Edit file", description: "Edit a local file" },
        { id: "travel", name: "Book flight", description: "Reserve air travel" },
      ]),
    ).resolves.toBeUndefined();
    expect((await catalog.searchAsync("edit a local file", 1))[0]?.skillId).toBe("edit");
  }, 60_000);
});

describe("loadEmbeddingArtifact", () => {
  it("shares one read across concurrent loads and caches the bytes for the process", async () => {
    const dir = mkdtempSync(
      fileURLToPath(new URL("../.embedding-artifact-test-", import.meta.url)),
    );
    dirs.push(dir);
    const path = join(dir, "fixture.rat1");
    writeFileSync(path, "RAT1-fixture");
    const [first, second] = await Promise.all([
      loadEmbeddingArtifact(path),
      loadEmbeddingArtifact(relative(REPO_ROOT, path)),
    ]);
    expect(Buffer.from(first).toString()).toBe("RAT1-fixture");
    expect(second).toBe(first);
    unlinkSync(path);
    expect(await loadEmbeddingArtifact(path)).toBe(first);
  });
});
