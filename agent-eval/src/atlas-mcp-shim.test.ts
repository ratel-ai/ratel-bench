import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  type AtlasTool,
  createShimServer,
  filterToolsForServer,
  httpSandbox,
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

describe("httpSandbox — the wire format against the real sandbox", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("POSTs /call-tool with {tool_name, tool_args}, not {name, arguments}", async () => {
    // The live sandbox's actual request schema — confirmed directly against it,
    // which 422s on {name, arguments} with "Field required" for both keys. This
    // went uncaught for a full build because httpSandbox had no test coverage
    // at all: every real tool CALL (not list) on both arms was silently
    // failing until this was caught in a live end-to-end run.
    const calls: Array<{ url: string; body: unknown }> = [];
    const fetchMock = vi.fn(async (url: string, init: RequestInit) => {
      calls.push({ url, body: JSON.parse(init.body as string) });
      return new Response(JSON.stringify({ content: [{ type: "text", text: "ok" }] }), {
        status: 200,
      });
    });
    vi.stubGlobal("fetch", fetchMock);

    await httpSandbox("http://localhost:1984").callTool("filesystem_list_allowed_directories", {
      foo: "bar",
    });

    expect(calls).toHaveLength(1);
    expect(calls[0].url).toBe("http://localhost:1984/call-tool");
    expect(calls[0].body).toEqual({
      tool_name: "filesystem_list_allowed_directories",
      tool_args: { foo: "bar" },
    });
  });

  it("throws with the response body on a non-2xx status", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response('{"detail":"bad request"}', { status: 422 })),
    );
    await expect(httpSandbox("http://localhost:1984").callTool("x", {})).rejects.toThrow(
      /HTTP 422/,
    );
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
