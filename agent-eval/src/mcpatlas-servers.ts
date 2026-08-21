// Catalog construction, per-arm config generation, and the tool-id normalization
// that lets the two arms be compared at all.

import { createHash } from "node:crypto";
import type {
  CanonicalToolId,
  McpAtlasArm,
  McpAtlasCatalogManifest,
  McpAtlasScope,
  McpAtlasServerSpec,
} from "./mcpatlas-types.js";

/** The 11 coding/data servers — the servers whose gold trajectories define the
 *  coding task set. Widening to `full` adds distractors, never gold. */
export const CODING_SERVERS = [
  "airtable",
  "cli-mcp-server",
  "context7",
  "desktop-commander",
  "e2b-server",
  "filesystem",
  "git",
  "github",
  "mcp-code-executor",
  "mcp-server-code-runner",
  "mongodb",
] as const;

/** Referenced by the dataset but absent from MCP-Atlas's own
 *  mcp_server_template.json, so they cannot be served: `full` is 195 tools / 36
 *  servers rather than the 220 / 40 the dataset implies. */
export const UNSERVABLE_SERVERS = [
  "anili",
  "balldontlie",
  "f1-mcp-server",
  "rijksmuseum-server",
] as const;

/** Env var NAMES per server, from mcp_server_template.json. Only three of the
 *  eleven coding servers need a credential; values never enter a manifest. */
export const SERVER_REQUIRED_ENV: Record<string, string[]> = {
  github: ["GITHUB_TOKEN"],
  airtable: ["AIRTABLE_API_KEY"],
  "e2b-server": ["E2B_API_KEY"],
  // Not visible in mcp_server_template.json — that file shows no `env` block for
  // mongodb — but the sandbox's own env.template requires it: the server talks to
  // YOUR Atlas cluster, seeded from data_exports/mongo_dump_video_game_store.
  // Missing it costs 10 tools and 11 tasks, and the failure would surface only
  // once cells started running.
  mongodb: ["MONGODB_CONNECTION_STRING"],
};

/** What ratel-local exposes here. `get_skill_content` needs a non-empty skill
 *  catalog and `auth` needs a runAuthFlow; this mode supplies neither. */
export const GATEWAY_TOOLS = ["search_capabilities", "invoke_tool"] as const;

// ─────────────────────────────────────────────────────────────────────────────
// Tool id normalization — the join key
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Reduce any of the three tool-id dialects to canonical `<server>/<tool>`.
 *
 *   MCP-Atlas dataset      `github_search_repositories`
 *   ratel-local telemetry  `github__search_repositories`
 *   Claude Code transcript `mcp__github__search_repositories`
 *
 * `knownServers` disambiguates by longest-prefix match, which matters because
 * tool names contain underscores freely while server names never do. Without
 * this the native arm scores 0 on every task by construction — its observed ids
 * would never textually match the dataset's gold ids.
 *
 * Returns `null` for anything unresolvable, so callers classify it as an
 * off-catalog call rather than coining a tool id that never existed.
 */
export function normalizeToolId(
  raw: string,
  knownServers: readonly string[],
): CanonicalToolId | null {
  if (!raw) return null;
  let s = raw.trim();
  if (s.includes("/")) {
    const [server, ...rest] = s.split("/");
    return knownServers.includes(server) && rest.length ? `${server}/${rest.join("/")}` : null;
  }
  if (s.startsWith("mcp__")) s = s.slice("mcp__".length);
  // Longest first, so a server that is a prefix of another cannot shadow it.
  const candidates = [...knownServers].sort((a, b) => b.length - a.length);
  for (const server of candidates) {
    for (const sep of ["__", "_"]) {
      const prefix = server + sep;
      if (s.startsWith(prefix)) {
        const tool = s.slice(prefix.length);
        if (tool) return `${server}/${tool}`;
      }
    }
  }
  return null;
}

export function serverOf(id: CanonicalToolId): string | null {
  const i = id.indexOf("/");
  return i > 0 ? id.slice(0, i) : null;
}

/** Canonical -> the `mcp__<server>__<tool>` name Claude Code reports and that
 *  `--allowedTools` expects. */
export function toClaudeToolName(id: CanonicalToolId): string {
  const i = id.indexOf("/");
  return `mcp__${id.slice(0, i)}__${id.slice(i + 1)}`;
}

/** Canonical -> the `<server>_<tool>` id the MCP-Atlas sandbox expects. */
export function toAtlasToolName(id: CanonicalToolId): string {
  const i = id.indexOf("/");
  return `${id.slice(0, i)}_${id.slice(i + 1)}`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Catalog manifests
// ─────────────────────────────────────────────────────────────────────────────

export function serversForScope(scope: McpAtlasScope, allServers: readonly string[]): string[] {
  const servable = allServers.filter((s) => !UNSERVABLE_SERVERS.includes(s as never));
  if (scope === "coding") return servable.filter((s) => CODING_SERVERS.includes(s as never)).sort();
  return [...servable].sort();
}

/** sha256 over sorted tool ids. Asserted equal across arms — a mismatch means
 *  the arms did not see the same universe and the comparison is void. */
export function catalogHash(toolIds: readonly CanonicalToolId[]): string {
  return createHash("sha256")
    .update([...toolIds].sort().join("\n"))
    .digest("hex");
}

export function buildCatalogManifest(
  scope: McpAtlasScope,
  toolsByServer: Record<string, CanonicalToolId[]>,
): McpAtlasCatalogManifest {
  const names = serversForScope(scope, Object.keys(toolsByServer));
  const servers: McpAtlasServerSpec[] = names.map((server) => {
    const tool_ids = [...(toolsByServer[server] ?? [])].sort();
    return {
      server,
      tool_ids,
      tool_count: tool_ids.length,
      required_env: SERVER_REQUIRED_ENV[server] ?? [],
    };
  });
  const all = servers.flatMap((s) => s.tool_ids);
  return {
    scope,
    servers,
    server_count: servers.length,
    tool_count: all.length,
    catalog_sha256: catalogHash(all),
  };
}

/** Env var names the manifest needs. Checked by `doctor` before any container
 *  starts, so a missing credential fails in seconds not mid-campaign. */
export function requiredEnv(manifest: McpAtlasCatalogManifest): string[] {
  return [...new Set(manifest.servers.flatMap((s) => s.required_env))].sort();
}

export function missingEnv(
  manifest: McpAtlasCatalogManifest,
  env: Record<string, string | undefined>,
): string[] {
  return requiredEnv(manifest).filter((k) => !env[k]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-arm configuration
// ─────────────────────────────────────────────────────────────────────────────

export interface ShimSpec {
  shimPath: string;
  sandboxUrl: string;
}

interface McpServerEntry {
  type?: string;
  command: string;
  args: string[];
  env?: Record<string, string>;
}

/** One shim per Atlas server — not one aggregated shim — so ratel-local registers
 *  N distinct upstreams and `search_capabilities` groups hits by server the way
 *  it does in production. */
export function buildShimEntries(
  manifest: McpAtlasCatalogManifest,
  shim: ShimSpec,
): Record<string, McpServerEntry> {
  const out: Record<string, McpServerEntry> = {};
  for (const s of manifest.servers) {
    out[s.server] = {
      type: "stdio",
      command: "node",
      args: [shim.shimPath, "--server", s.server, "--sandbox-url", shim.sandboxUrl],
    };
  }
  return out;
}

export function buildNativeMcpConfig(
  manifest: McpAtlasCatalogManifest,
  shim: ShimSpec,
): { mcpServers: Record<string, McpServerEntry> } {
  return { mcpServers: buildShimEntries(manifest, shim) };
}

/** ratel-local's own config: the identical upstream set the native arm used,
 *  passed as an explicit path so `~/.ratel` is never consulted. */
export function buildRatelServeConfig(
  manifest: McpAtlasCatalogManifest,
  shim: ShimSpec,
  retrievalMethod: "bm25" | "semantic" | "hybrid",
): Record<string, unknown> {
  const cfg: Record<string, unknown> = { mcpServers: buildShimEntries(manifest, shim) };
  // The gateway treats an absent method as bm25; keep the config minimal so the
  // default path is exercised exactly as a user would get it.
  if (retrievalMethod !== "bm25") cfg.retrieval = { method: retrievalMethod };
  return cfg;
}

export function buildRatelMcpConfig(opts: {
  ratelLocalPin: string;
  serveConfigPath: string;
  telemetryPath: string;
  env?: Record<string, string>;
}): { mcpServers: Record<string, McpServerEntry> } {
  return {
    mcpServers: {
      "ratel-local": {
        type: "stdio",
        command: "npx",
        args: [
          "-y",
          `@ratel-ai/ratel-local@${opts.ratelLocalPin}`,
          "serve",
          opts.serveConfigPath,
          "--telemetry-file",
          opts.telemetryPath,
        ],
        ...(opts.env ? { env: opts.env } : {}),
      },
    },
  };
}

/** `--allowedTools` per arm. native gets the whole catalog; ratel gets only the
 *  gateway surface — which is what makes the context-size difference real. */
export function allowedToolsFor(arm: McpAtlasArm, manifest: McpAtlasCatalogManifest): string[] {
  if (arm === "ratel") return GATEWAY_TOOLS.map((t) => `mcp__ratel-local__${t}`);
  return manifest.servers.flatMap((s) => s.tool_ids.map(toClaudeToolName)).sort();
}
