/**
 * MetaTool → Dataset ingest (Lane B: data-only, our own loop/grader).
 *
 * Source: https://github.com/HowieHwong/MetaTool (MIT). We pull two upstream
 * files and normalize the SINGLE-tool slice into the harness `Dataset` shape:
 *   - dataset/plugin_des.json        — { plugin_name: description } over 199 plugins
 *   - dataset/data/all_clean_data.csv — single-tool queries, columns `Query,Tool`
 *
 * Mapping (faithful to the old Rust ingest):
 *   - tool universe = plugin_des.json; id == name == plugin key; empty schema.
 *   - each query → a Scenario with ONE turn; id `metatool-st-<file-line>`.
 *   - expectedTool = the gold plugin; expectedQuery = the query itself
 *     (MetaTool ships no curated query, so input-only == expected-query here).
 *   - queries whose gold tool is absent from plugin_des.json are skipped.
 *
 * The multi-tool slice is intentionally NOT ingested: a Turn carries a single
 * expectedTool, so the N-gold queries don't map without changing the schema.
 *
 * Run: `pnpm ingest:metatool` (uses cache) or `pnpm ingest:metatool --download`
 * (force re-fetch). Raw files cache under datasets/.cache/metatool/.
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import type { Dataset, Scenario, ToolSpec } from "../types.js";

const PLUGIN_DES_URL =
  "https://raw.githubusercontent.com/HowieHwong/MetaTool/master/dataset/plugin_des.json";
const SINGLE_TOOL_CSV_URL =
  "https://raw.githubusercontent.com/HowieHwong/MetaTool/master/dataset/data/all_clean_data.csv";

const CACHE_DIR = "datasets/.cache/metatool";
const OUT_PATH = "datasets/metatool.json";

async function cachedFetch(url: string, dest: string, force: boolean): Promise<string> {
  if (!force && existsSync(dest)) return readFileSync(dest, "utf-8");
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url} -> ${res.status} ${res.statusText}`);
  const text = await res.text();
  mkdirSync(dirname(dest), { recursive: true });
  writeFileSync(dest, text);
  return text;
}

function parsePlugins(json: string): Map<string, ToolSpec> {
  const raw = JSON.parse(json) as Record<string, string>;
  const out = new Map<string, ToolSpec>();
  for (const [name, description] of Object.entries(raw)) {
    out.set(name, { id: name, name, description: String(description).trim(), inputSchema: {} });
  }
  return out;
}

interface RawRow {
  line: number;
  query: string;
  tool: string;
}

/**
 * Parse `Query,Tool` line-by-line. The Tool column is the LAST field and never
 * contains a comma, so splitting on the last comma is robust to unquoted commas
 * in queries; quoted queries get their wrapping quotes stripped. The single
 * malformed upstream row (unterminated quote) yields a non-plugin "tool" and is
 * dropped downstream as unknown-gold.
 */
function parseSingleTool(csv: string): RawRow[] {
  const lines = csv.split(/\r?\n/);
  const out: RawRow[] = [];
  for (let i = 1; i < lines.length; i++) {
    const raw = lines[i];
    if (!raw) continue;
    const c = raw.lastIndexOf(",");
    if (c < 0) continue;
    const tool = raw.slice(c + 1).trim();
    let query = raw.slice(0, c).trim();
    if (query.length >= 2 && query.startsWith('"') && query.endsWith('"')) {
      query = query.slice(1, -1).replace(/""/g, '"');
    }
    if (!query || !tool) continue;
    out.push({ line: i + 1, query, tool }); // header is file line 1
  }
  return out;
}

function byId(a: { id: string }, b: { id: string }): number {
  return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
}

function build(
  plugins: Map<string, ToolSpec>,
  rows: RawRow[],
): { scenarios: Scenario[]; skipped: number } {
  const scenarios: Scenario[] = [];
  let skipped = 0;
  for (const r of rows) {
    if (!plugins.has(r.tool)) {
      skipped++;
      continue;
    }
    scenarios.push({
      id: `metatool-st-${r.line}`,
      turns: [
        {
          input: { messages: [{ role: "user", content: r.query }] },
          expectedTool: r.tool,
          expectedQuery: r.query,
        },
      ],
    });
  }
  scenarios.sort(byId); // stable id-sorted output keeps re-ingest diffs readable
  return { scenarios, skipped };
}

async function main(): Promise<void> {
  const force = process.argv.includes("--download");
  const pluginsJson = await cachedFetch(PLUGIN_DES_URL, `${CACHE_DIR}/plugin_des.json`, force);
  const csv = await cachedFetch(SINGLE_TOOL_CSV_URL, `${CACHE_DIR}/all_clean_data.csv`, force);

  const plugins = parsePlugins(pluginsJson);
  const rows = parseSingleTool(csv);
  const { scenarios, skipped } = build(plugins, rows);

  const dataset: Dataset = { tools: [...plugins.values()].sort(byId), scenarios };
  mkdirSync(dirname(OUT_PATH), { recursive: true });
  writeFileSync(OUT_PATH, JSON.stringify(dataset));

  console.error(
    `plugins=${plugins.size} rows=${rows.length} scenarios=${scenarios.length} ` +
      `skipped_unknown_gold=${skipped}`,
  );
  console.error(`wrote ${OUT_PATH}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
