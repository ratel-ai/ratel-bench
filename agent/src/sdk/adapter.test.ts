import type { ExecutableTool, Skill } from "@ratel-ai/sdk";
import { afterEach, describe, expect, it } from "vitest";
import { buildSkillCatalog, buildToolCatalog, gatewayTools } from "./adapter.js";
import { resetForTests, sdkVersion, selectVersion } from "./resolve.js";

const tools: ExecutableTool[] = [
  {
    id: "fs__read_file",
    name: "read_file",
    description: "Read a file from disk",
    inputSchema: { type: "object", properties: { path: { type: "string" } } },
    outputSchema: { type: "object" },
    execute: async ({ path }) => ({ contents: `read ${path}` }),
  },
  {
    id: "mail__send_email",
    name: "send_email",
    description: "Send an email message",
    inputSchema: { type: "object" },
    outputSchema: { type: "object" },
    execute: async () => ({ sent: true }),
  },
];

const skills: Skill[] = [
  { id: "edit-file", name: "Edit file", description: "Edit a local file", body: "edit steps" },
  { id: "book-flight", name: "Book flight", description: "Reserve air travel", body: "book steps" },
];

afterEach(resetForTests);

describe.each(["0.2.0", "0.4.0", "0.5.0", "0.12.0"])("SDK %s offline contract", (version) => {
  it("registers and retrieves the intended tool and skill", async () => {
    selectVersion(version);
    expect(sdkVersion()).toBe(version);
    const toolCatalog = await buildToolCatalog({ tools });
    const skillCatalog = await buildSkillCatalog({ skills });
    expect((await toolCatalog.search("read a file", 1))[0]?.toolId).toBe("fs__read_file");
    expect((await skillCatalog.search("edit a file", 1))[0]?.skillId).toBe("edit-file");
  });

  it("discovers and invokes through the gateway", async () => {
    selectVersion(version);
    const { catalog } = await buildToolCatalog({ tools });
    const { searchToolsTool, invokeToolTool } = await gatewayTools();
    const found = await searchToolsTool(catalog).execute({ query: "read a file", topK: 1 });
    expect(JSON.stringify(found)).toContain("fs__read_file");
    const invoked = await invokeToolTool(catalog).execute({
      toolId: "fs__read_file",
      args: { path: "notes.txt" },
    });
    expect(JSON.stringify(invoked)).toContain("read notes.txt");
  });
});
