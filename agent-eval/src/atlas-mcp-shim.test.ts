import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { describe, expect, it } from "vitest";
import {
  type AtlasTool,
  createShimServer,
  filterToolsForServer,
  qualify,
  type SandboxClient,
  toToolResult,
} from "./atlas-mcp-shim.js";

const ALL: AtlasTool[] = [
  { name: "github_search_repositories", description: "search repos" },
  { name: "github_get_issue", description: "get issue" },
  { name: "git_status", description: "status" },
  { name: "weather_get_forecast", description: "forecast" },
];

function stubSandbox(over: Partial<SandboxClient> = {}): SandboxClient & { calls: string[] } {
  const calls: string[] = [];
  return {
    calls,
    async listTools() {
      return ALL;
    },
    async callTool(name, args) {
      calls.push(name);
      return { content: [{ type: "text", text: `ran ${name} ${JSON.stringify(args)}` }] };
    },
    ...over,
  } as SandboxClient & { calls: string[] };
}

describe("filterToolsForServer", () => {
  it("keeps only this server's tools and strips the prefix", () => {
    expect(filterToolsForServer(ALL, "github").map((t) => t.name)).toEqual([
      "search_repositories",
      "get_issue",
    ]);
  });

  it("does not let `git` capture `github` tools", () => {
    expect(filterToolsForServer(ALL, "git").map((t) => t.name)).toEqual(["status"]);
  });

  it("supplies an object inputSchema when the sandbox omits one", () => {
    expect(filterToolsForServer(ALL, "git")[0].inputSchema).toEqual({
      type: "object",
      properties: {},
    });
  });

  it("preserves a provided schema", () => {
    const schema = { type: "object", properties: { q: { type: "string" } } };
    const [t] = filterToolsForServer([{ name: "git_x", inputSchema: schema }], "git");
    expect(t.inputSchema).toBe(schema);
  });

  it("drops the sandbox's explicit-null fields rather than passing them through", () => {
    // The real sandbox emits title/outputSchema/annotations/_meta as literal
    // `null`, not absent. Claude Code's and ratel-local's MCP client schemas
    // reject `null` for these optional fields — spread-through nulls made
    // every server fail tools/list validation and lose its connection after
    // ~12s of retries, leaving the agent with zero tools on both arms.
    const raw = [
      {
        name: "git_status",
        title: null,
        description: "status",
        inputSchema: { type: "object", properties: {} },
        outputSchema: null,
        annotations: null,
        _meta: null,
      },
    ] as unknown as AtlasTool[];
    const [t] = filterToolsForServer(raw, "git");
    expect(Object.values(t)).not.toContain(null);
    expect(t).toEqual({
      name: "status",
      description: "status",
      inputSchema: { type: "object", properties: {} },
    });
  });

  it("omits description entirely rather than passing through null/undefined", () => {
    const raw = [{ name: "git_status", description: null }] as unknown as AtlasTool[];
    const [t] = filterToolsForServer(raw, "git");
    expect("description" in t).toBe(false);
  });

  it("re-qualifies on the way back out", () => {
    expect(qualify("github", "get_issue")).toBe("github_get_issue");
  });
});

describe("toToolResult", () => {
  it("passes through MCP-shaped content", () => {
    expect(toToolResult({ content: [{ type: "text", text: "hi" }] })).toEqual({
      content: [{ type: "text", text: "hi" }],
      isError: false,
    });
  });

  it("marks declared errors", () => {
    expect(toToolResult({ error: "nope" })).toEqual({
      content: [{ type: "text", text: "nope" }],
      isError: true,
    });
  });

  it("stringifies anything else", () => {
    expect(toToolResult({ a: 1 }).content[0].text).toBe('{"a":1}');
    expect(toToolResult(null).content[0].text).toBe("null");
  });
});

describe("shim contract over a real MCP transport", () => {
  async function connect(sandbox: SandboxClient) {
    const server = createShimServer("github", sandbox);
    const [clientT, serverT] = InMemoryTransport.createLinkedPair();
    const client = new Client({ name: "test", version: "0" }, { capabilities: {} });
    await Promise.all([server.connect(serverT), client.connect(clientT)]);
    return client;
  }

  it("tools/list returns exactly this server's tools, prefixes stripped", async () => {
    const client = await connect(stubSandbox());
    const { tools } = await client.listTools();
    expect(tools.map((t) => t.name)).toEqual(["search_repositories", "get_issue"]);
  });

  it("tools/call round-trips and re-qualifies the name upstream", async () => {
    const sandbox = stubSandbox();
    const client = await connect(sandbox);
    const res = await client.callTool({ name: "get_issue", arguments: { id: 7 } });
    expect(sandbox.calls).toEqual(["github_get_issue"]);
    expect((res.content as Array<{ text: string }>)[0].text).toContain("github_get_issue");
  });

  it("an upstream throw becomes a structured error result, not a protocol crash", async () => {
    const sandbox = stubSandbox({
      async callTool() {
        throw new Error("upstream exploded");
      },
    });
    const client = await connect(sandbox);
    const res = await client.callTool({ name: "get_issue", arguments: {} });
    expect(res.isError).toBe(true);
    expect((res.content as Array<{ text: string }>)[0].text).toContain("upstream exploded");
  });
});
