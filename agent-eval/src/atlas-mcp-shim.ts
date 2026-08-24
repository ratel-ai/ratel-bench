// HTTP -> MCP adapter for the MCP-Atlas sandbox.
//
// The sandbox speaks plain HTTP (`POST /list-tools`, `POST /call-tool` on :1984),
// not MCP. Neither Claude Code nor ratel-local can consume that, so every arm
// goes through this shim.
//
// One process per Atlas server, not one aggregated process. That matters for the
// ratel arm: ratel-local must register N distinct upstreams so `search_tools`
// groups hits by server the way it does in production, and so
// `ratel_tool_payload` telemetry reports per-server schema token counts.

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { CallToolRequestSchema, ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types.js";

export interface AtlasTool {
  name: string;
  description?: string;
  inputSchema?: Record<string, unknown>;
}

/** Tools belonging to `server`, with the `<server>_` prefix stripped so the tool
 *  looks native to the agent. Restored on the way back out in `callTool`.
 *
 * Rebuilds each tool as a clean `{name, description, inputSchema}` rather than
 * spreading the sandbox's raw object through: the sandbox emits explicit
 * `null` for optional fields it doesn't use (`title`, `outputSchema`,
 * `annotations`, `_meta`), and the MCP client's tools/list schema — both
 * Claude Code's and ratel-local's — treats a `null` there as an invalid value,
 * not an absent one. Passed through verbatim, every one of these tools failed
 * validation and every server's connection was torn down after retrying for
 * ~12s, leaving the agent with zero tools on both arms. */
export function filterToolsForServer(tools: AtlasTool[], server: string): AtlasTool[] {
  const prefix = `${server}_`;
  return tools
    .filter((t) => typeof t?.name === "string" && t.name.startsWith(prefix))
    .map((t) => ({
      name: t.name.slice(prefix.length),
      ...(t.description ? { description: t.description } : {}),
      // The MCP spec requires an object schema; the sandbox sometimes omits it.
      inputSchema: t.inputSchema ?? { type: "object", properties: {} },
    }));
}

export function qualify(server: string, bare: string): string {
  return `${server}_${bare}`;
}

export interface SandboxClient {
  listTools(): Promise<AtlasTool[]>;
  callTool(name: string, args: Record<string, unknown>): Promise<unknown>;
}

/** Minimal client for the sandbox's two endpoints. Kept behind an interface so
 *  the shim's behaviour can be tested without a live sandbox. */
export function httpSandbox(sandboxUrl: string, timeoutMs = 60_000): SandboxClient {
  const post = async (path: string, body: unknown): Promise<unknown> => {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), timeoutMs);
    try {
      const res = await fetch(`${sandboxUrl.replace(/\/+$/, "")}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: ctrl.signal,
      });
      if (!res.ok) throw new Error(`${path} -> HTTP ${res.status} ${await res.text()}`);
      return await res.json();
    } finally {
      clearTimeout(t);
    }
  };
  return {
    async listTools() {
      const body = (await post("/list-tools", {})) as { tools?: AtlasTool[] } | AtlasTool[];
      return Array.isArray(body) ? body : (body.tools ?? []);
    },
    async callTool(name, args) {
      // The sandbox's actual request schema is {tool_name, tool_args} — verified
      // directly against the live sandbox, which 422s on {name, arguments} with
      // "Field required" for both. This was never exercised until a real
      // end-to-end run: every /call-tool request on both arms was silently
      // failing, and the native arm's failure classification (below) had no
      // way to surface it, since it never touches ratel's telemetry.
      return await post("/call-tool", { tool_name: name, tool_args: args });
    },
  };
}

/** Normalize whatever the sandbox returns into MCP tool-result content. */
export function toToolResult(payload: unknown): {
  content: Array<{ type: "text"; text: string }>;
  isError?: boolean;
} {
  if (payload && typeof payload === "object") {
    const o = payload as Record<string, unknown>;
    if (typeof o.error === "string") {
      return { content: [{ type: "text", text: o.error }], isError: true };
    }
    if (Array.isArray(o.content)) {
      const text = o.content
        .map((c) => {
          const cc = c as Record<string, unknown>;
          return typeof cc?.text === "string" ? cc.text : JSON.stringify(c);
        })
        .join("\n");
      return { content: [{ type: "text", text }], isError: Boolean(o.isError) };
    }
  }
  return { content: [{ type: "text", text: JSON.stringify(payload ?? null) }] };
}

export function createShimServer(server: string, sandbox: SandboxClient): Server {
  const mcp = new Server(
    { name: `atlas-${server}`, version: "0.0.0" },
    { capabilities: { tools: {} } },
  );

  mcp.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: filterToolsForServer(await sandbox.listTools(), server),
  }));

  mcp.setRequestHandler(CallToolRequestSchema, async (req) => {
    const bare = req.params.name;
    const args = (req.params.arguments ?? {}) as Record<string, unknown>;
    try {
      return toToolResult(await sandbox.callTool(qualify(server, bare), args));
    } catch (err) {
      // A structured error result, not a protocol-level throw: the agent should
      // see a failed tool call and be able to recover, and the benchmark needs to
      // classify the failure rather than lose the whole cell.
      return {
        content: [{ type: "text", text: `${bare} failed: ${(err as Error).message}` }],
        isError: true,
      };
    }
  });

  return mcp;
}

function arg(name: string, fallback: string): string {
  const i = process.argv.indexOf(name);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

async function main(): Promise<void> {
  const server = arg("--server", "");
  const sandboxUrl = arg("--sandbox-url", process.env.MCP_SANDBOX_URL ?? "http://localhost:1984");
  if (!server) throw new Error("atlas-mcp-shim requires --server <name>");
  const mcp = createShimServer(server, httpSandbox(sandboxUrl));
  await mcp.connect(new StdioServerTransport());
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((err) => {
    // stderr only — stdout is the MCP transport.
    process.stderr.write(`atlas-mcp-shim: ${(err as Error).message}\n`);
    process.exit(1);
  });
}
