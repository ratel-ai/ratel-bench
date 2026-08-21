// MCP-Atlas public split -> pinned corpus + catalog manifests.
//
// This stage is the gate: it asserts the task count and that every gold tool is
// inside the catalog, and THROWS before the campaign can spend anything. The
// gold-outside-catalog check exists because that exact contamination silently
// depressed the 0.4.0 SR-Agents numbers.

import { createHash } from "node:crypto";
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import {
  buildCatalogManifest,
  CODING_SERVERS,
  normalizeToolId,
  serversForScope,
  UNSERVABLE_SERVERS,
} from "./mcpatlas-servers.js";
import type {
  CanonicalToolId,
  McpAtlasCatalogManifest,
  McpAtlasScope,
  McpAtlasTask,
} from "./mcpatlas-types.js";

/** Expected size of the coding task set. A drift here means the dataset moved and
 *  every published number is on a different corpus — hence a hard failure. */
export const EXPECTED_TASK_COUNT = 55;
/** Expected `coding` catalog size. */
export const EXPECTED_CODING_TOOLS = 79;

/** A raw MCP-Atlas row as published (HF datasets-server or JSONL). */
export interface RawAtlasRow {
  TASK: string;
  PROMPT: string;
  ENABLED_TOOLS: unknown;
  GTFA_CLAIMS: unknown;
  TRAJECTORY: unknown;
}

/**
 * MCP-Atlas ships some list-valued columns as Python reprs rather than JSON, and
 * some ENABLED_TOOLS entries as objects rather than strings. Both are silent
 * traps: a JSON-only parser reports those rows as empty, which is how a 55-task
 * set first looked like 53.
 */
export function parseListField(v: unknown): unknown[] {
  if (Array.isArray(v)) return v;
  if (typeof v !== "string") return [];
  const s = v.trim();
  if (!s) return [];
  try {
    const p = JSON.parse(s);
    return Array.isArray(p) ? p : [];
  } catch {
    // Python repr: single-quoted strings, True/False/None.
    try {
      const jsonish = s
        .replace(/(?<![\\])'/g, '"')
        .replace(/\bTrue\b/g, "true")
        .replace(/\bFalse\b/g, "false")
        .replace(/\bNone\b/g, "null");
      const p = JSON.parse(jsonish);
      return Array.isArray(p) ? p : [];
    } catch {
      return [];
    }
  }
}

/** Pull tool names out of ENABLED_TOOLS, which mixes plain strings and objects. */
export function toolNamesFrom(v: unknown): string[] {
  const out: string[] = [];
  for (const x of parseListField(v)) {
    if (typeof x === "string") out.push(x);
    else if (x && typeof x === "object") {
      const o = x as Record<string, unknown>;
      const n = o.name ?? o.tool ?? o.toolId;
      if (typeof n === "string") out.push(n);
    }
  }
  return out;
}

export interface GoldCall {
  step: number;
  tool_id: CanonicalToolId;
  args: Record<string, unknown>;
  recorded_output_excerpt: string;
}

/** Gold tool calls from the recorded TRAJECTORY, with the upstream output that
 *  followed each one. The outputs are what make a replay harness possible later;
 *  we keep an excerpt so per-task reports can show them without bloating rows. */
export function goldCallsFrom(
  trajectory: unknown,
  knownServers: readonly string[],
  excerptChars = 400,
): GoldCall[] {
  const msgs = parseListField(trajectory);
  const calls: GoldCall[] = [];
  let step = 0;
  for (let i = 0; i < msgs.length; i++) {
    const m = msgs[i] as Record<string, unknown> | undefined;
    if (!m || typeof m !== "object") continue;
    const tcs = m.tool_calls;
    if (!Array.isArray(tcs)) continue;
    for (const tc of tcs) {
      const fn = (tc as Record<string, unknown>)?.function as Record<string, unknown> | undefined;
      const name = fn?.name;
      if (typeof name !== "string") continue;
      const id = normalizeToolId(name, knownServers);
      if (!id) continue;
      let args: Record<string, unknown> = {};
      if (typeof fn?.arguments === "string") {
        try {
          args = JSON.parse(fn.arguments as string);
        } catch {
          args = { _raw: fn.arguments };
        }
      } else if (fn?.arguments && typeof fn.arguments === "object") {
        args = fn.arguments as Record<string, unknown>;
      }
      // The tool result is the next message with role "tool".
      let excerpt = "";
      for (let j = i + 1; j < msgs.length; j++) {
        const n = msgs[j] as Record<string, unknown>;
        if (n?.role === "tool") {
          excerpt = JSON.stringify(n.content ?? "").slice(0, excerptChars);
          break;
        }
        if (n?.role === "assistant") break;
      }
      calls.push({ step: step++, tool_id: id, args, recorded_output_excerpt: excerpt });
    }
  }
  return calls;
}

/** Distinct servers a task's gold trajectory touches. */
export function goldServers(calls: GoldCall[]): string[] {
  return [...new Set(calls.map((c) => c.tool_id.split("/")[0]))].sort();
}

/** Every tool id the dataset ever exposes, grouped by server. This is the tool
 *  universe: 220 across 40 servers, of which 4 servers are unservable. */
export function toolsByServerFrom(
  rows: RawAtlasRow[],
  knownServers: readonly string[],
): Record<string, CanonicalToolId[]> {
  const out: Record<string, Set<CanonicalToolId>> = {};
  for (const r of rows) {
    for (const name of toolNamesFrom(r.ENABLED_TOOLS)) {
      const id = normalizeToolId(name, knownServers);
      if (!id) continue;
      const server = id.split("/")[0];
      if (!out[server]) out[server] = new Set();
      out[server].add(id);
    }
  }
  return Object.fromEntries(Object.entries(out).map(([k, v]) => [k, [...v].sort()]));
}

/** Server names as they appear in the raw tool ids, before canonicalization. */
export function discoverServers(rows: RawAtlasRow[]): string[] {
  const s = new Set<string>();
  for (const r of rows) {
    for (const name of toolNamesFrom(r.ENABLED_TOOLS)) {
      const head = name.split("_")[0];
      if (head) s.add(head);
    }
  }
  return [...s].sort();
}

export function toTask(row: RawAtlasRow, knownServers: readonly string[]): McpAtlasTask {
  const gold_calls = goldCallsFrom(row.TRAJECTORY, knownServers);
  const gold_tool_ids = [...new Set(gold_calls.map((c) => c.tool_id))];
  const enabled_tool_ids = toolNamesFrom(row.ENABLED_TOOLS)
    .map((n) => normalizeToolId(n, knownServers))
    .filter((x): x is CanonicalToolId => x !== null);
  return {
    id: `mcpatlas-${row.TASK}`,
    task_id: row.TASK,
    prompt: row.PROMPT,
    enabled_tool_ids: [...new Set(enabled_tool_ids)].sort(),
    gold_tool_ids,
    gold_servers: goldServers(gold_calls),
    gold_calls,
    claims: parseListField(row.GTFA_CLAIMS).filter((c): c is string => typeof c === "string"),
  };
}

/**
 * The coding task set: tasks whose gold trajectory touches ONLY coding/data
 * servers.
 *
 * The stricter-than-obvious rule is deliberate. 303 of 500 tasks *touch* a coding
 * server, but 248 of those are cross-domain lookups where the coding server is
 * incidental — `github` used as a data source about a repository rather than as a
 * development tool. Selecting on "touches" would trade a coding benchmark for a
 * general one.
 */
export function selectCodingTasks(tasks: McpAtlasTask[]): McpAtlasTask[] {
  const coding = new Set<string>(CODING_SERVERS);
  return tasks
    .filter((t) => t.gold_servers.length > 0 && t.gold_servers.every((s) => coding.has(s)))
    .sort((a, b) => a.task_id.localeCompare(b.task_id));
}

/** sha256 over sorted task ids. Pinned in the frozen config; a change means a
 *  different corpus, and the campaign refuses to append across one. */
export function taskListHash(tasks: McpAtlasTask[]): string {
  return createHash("sha256")
    .update(
      tasks
        .map((t) => t.task_id)
        .sort()
        .join("\n"),
    )
    .digest("hex");
}

export interface CoverageReport {
  total: number;
  covered: number;
  uncovered: Array<{ task_id: string; missing: CanonicalToolId[] }>;
}

/** gold ⊆ catalog, per task. */
export function goldCoverage(
  tasks: McpAtlasTask[],
  manifest: McpAtlasCatalogManifest,
): CoverageReport {
  const cat = new Set(manifest.servers.flatMap((s) => s.tool_ids));
  const uncovered: CoverageReport["uncovered"] = [];
  for (const t of tasks) {
    const missing = t.gold_tool_ids.filter((g) => !cat.has(g));
    if (missing.length) uncovered.push({ task_id: t.task_id, missing });
  }
  return { total: tasks.length, covered: tasks.length - uncovered.length, uncovered };
}

// ─────────────────────────────────────────────────────────────────────────────
// I/O
// ─────────────────────────────────────────────────────────────────────────────

/** Read a directory of HF datasets-server pages (`rows_*.json`) or a JSONL. */
export function loadRows(path: string): RawAtlasRow[] {
  const p = resolve(path);
  const out: RawAtlasRow[] = [];
  const readJsonPage = (file: string): void => {
    const d = JSON.parse(readFileSync(file, "utf8"));
    for (const r of d.rows ?? []) out.push(r.row as RawAtlasRow);
  };
  if (p.endsWith(".jsonl")) {
    for (const line of readFileSync(p, "utf8").split("\n")) {
      if (line.trim()) out.push(JSON.parse(line) as RawAtlasRow);
    }
  } else if (p.endsWith(".json")) {
    readJsonPage(p);
  } else {
    for (const f of readdirSync(p).sort()) {
      if (f.startsWith("rows_") && f.endsWith(".json")) readJsonPage(join(p, f));
    }
  }
  return out;
}

function arg(name: string, fallback: string): string {
  const i = process.argv.indexOf(name);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

export function main(): void {
  const rowsPath = arg("--rows", "fixtures/mcpatlas/rows");
  const outDir = arg("--out", "fixtures/mcpatlas");
  const corpusOut = arg("--corpus-out", "test-data/mcpatlas-coding.jsonl");
  const datasetRevision = arg("--dataset-revision", "unpinned");

  const rows = loadRows(rowsPath);
  if (!rows.length) throw new Error(`no MCP-Atlas rows found at ${rowsPath}`);

  const servers = discoverServers(rows);
  const toolsByServer = toolsByServerFrom(rows, servers);
  const universe = Object.values(toolsByServer).flat().length;

  const tasks = selectCodingTasks(rows.map((r) => toTask(r, servers)));
  const manifests: Record<McpAtlasScope, McpAtlasCatalogManifest> = {
    coding: buildCatalogManifest("coding", toolsByServer),
    full: buildCatalogManifest("full", toolsByServer),
  };

  // ── the gate ───────────────────────────────────────────────────────────────
  const problems: string[] = [];
  if (tasks.length !== EXPECTED_TASK_COUNT) {
    problems.push(`expected ${EXPECTED_TASK_COUNT} coding tasks, got ${tasks.length}`);
  }
  if (manifests.coding.tool_count !== EXPECTED_CODING_TOOLS) {
    problems.push(
      `expected ${EXPECTED_CODING_TOOLS} tools in the coding catalog, got ${manifests.coding.tool_count}`,
    );
  }
  const cov = goldCoverage(tasks, manifests.coding);
  if (cov.uncovered.length) {
    const detail = cov.uncovered
      .slice(0, 5)
      .map((u) => `${u.task_id}: ${u.missing.join(", ")}`)
      .join("; ");
    problems.push(
      `${cov.uncovered.length}/${cov.total} tasks have gold tools outside the coding catalog ` +
        `(e.g. ${detail}). Both arms would fail those tasks and their retrieval metrics ` +
        `would be meaningless — widen the catalog or narrow the task set.`,
    );
  }
  if (problems.length) {
    throw new Error(`mcpatlas ingest refuses to proceed:\n  - ${problems.join("\n  - ")}`);
  }

  // ── write ──────────────────────────────────────────────────────────────────
  mkdirSync(outDir, { recursive: true });
  mkdirSync(resolve(corpusOut, ".."), { recursive: true });
  writeFileSync(corpusOut, `${tasks.map((t) => JSON.stringify(t)).join("\n")}\n`);
  const hash = taskListHash(tasks);
  writeFileSync(
    join(outDir, "tasks-coding-v1.json"),
    `${JSON.stringify(
      {
        dataset_revision: datasetRevision,
        task_list_hash: hash,
        task_count: tasks.length,
        task_ids: tasks.map((t) => t.task_id).sort(),
      },
      null,
      2,
    )}\n`,
  );
  for (const scope of ["coding", "full"] as const) {
    writeFileSync(
      join(outDir, `catalog-${scope}.json`),
      `${JSON.stringify(manifests[scope], null, 2)}\n`,
    );
  }

  console.log(
    [
      `tasks           ${tasks.length}`,
      `task_list_hash  ${hash.slice(0, 16)}`,
      `tool universe   ${universe} across ${servers.length} servers`,
      `unservable      ${UNSERVABLE_SERVERS.filter((s) => servers.includes(s)).join(", ") || "none"}`,
      `catalog coding  ${manifests.coding.tool_count} tools / ${manifests.coding.server_count} servers`,
      `catalog full    ${manifests.full.tool_count} tools / ${manifests.full.server_count} servers`,
      `gold coverage   ${cov.covered}/${cov.total}`,
      `servable scope  ${serversForScope("coding", servers).join(", ")}`,
    ].join("\n"),
  );
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}
