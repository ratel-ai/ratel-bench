// `results-audit` — read-only health check of raw agent JSONL (BFCL cells or
// SR-Agents selection cells) before summarizing or migrating it:
//   - counts rows by file / label / arm / model / error class / ratel_version
//   - lists the control keys the OLD control cache (earliest row wins, errors
//     included) would serve as errors from the given cache file set
//   - flags label × arm × model groups that mix providers or output caps
//   - BFCL: lists label cells whose rows span >1 `ratel_version` (the supersede
//     key includes it, so each version counts as its own cell), and per label the
//     `ratel_version` of its infra-errored rows: the `--ratel-version` a re-drain
//     must stamp to supersede them
//
//   results-audit --bench bfcl|sragents --agent P [--cache P,…] [--drop-infra-errors --out P]
//
// `--drop-infra-errors --out P` writes the agent file to P minus the
// `transient|access` rows a later row of the same cell already supersedes (the
// summarizers' key and rule), so the cleanup never changes a summary: final infra
// rows stay (still counted in `excluded_cells`), and so does a superseded one
// newer than its cell's kept row (it keeps the group timestamp). Every other
// non-blank line is kept byte-for-byte; blank lines are dropped. Nothing is
// written unless `--out` is given.

import { readFileSync, writeFileSync } from "node:fs";
import { basename, resolve } from "node:path";
import {
  type ErrorClass,
  type ErrorRow,
  errorClassOf,
  isInfraError,
  supersede,
} from "./cell-errors.js";
import { resolveRepoPath } from "./paths.js";
import {
  isTruncated,
  labelledCellKeyOf,
  type VersionSplitCell,
  versionSplitCells,
} from "./report.js";
import { sragentsCellKeyOf } from "./sragents-summarize.js";
import type { SragentsSelectCell } from "./sragents-types.js";
import type { CellResult } from "./types.js";

export type Bench = "bfcl" | "sragents";

/** The row fields the audit reads; BFCL `CellResult` and SR `SragentsSelectCell` both carry them. */
export interface AuditRow extends ErrorRow {
  scenario_id: string;
  arm: string;
  model: string;
  run_index: number;
  pool_size: number | null;
  generated_at?: string;
  /** The label reports group on. */
  ratel_ai_core_version?: string;
  /** `@ratel-ai/sdk` version (BFCL only). */
  ratel_version?: string;
  provider?: string;
  /** Requested output cap; absent on rows written before caps were recorded. */
  max_output_tokens?: number | null;
  truncated_steps?: number;
  finish_reason?: string;
}

export interface AuditFile {
  file: string;
  rows: AuditRow[];
}

export interface AuditCount {
  file: string;
  label: string;
  arm: string;
  model: string;
  error_class: ErrorClass | "ok";
  ratel_version: string;
  rows: number;
  /** Rows cut off by the output-token limit (`truncated_steps>0` or finish `length`). */
  truncated: number;
}

/** A control key whose OLD-rule cache winner is an errored row. */
export interface StaleControl {
  key: string;
  arm: string;
  model: string;
  label: string;
  error_class: ErrorClass;
  error: string;
  file: string;
  generated_at: string;
}

export interface MixedHarness {
  label: string;
  arm: string;
  model: string;
  /** Row count per provider (`unset` when not recorded). */
  providers: Record<string, number>;
  /** Row count per requested output cap (`unset` when not recorded). */
  caps: Record<string, number>;
}

/** Arms the OLD control cache served; a frozen copy of the pre-U2 rule the audit replays. */
const CACHEABLE_ARMS: ReadonlySet<string> = new Set(["control-baseline", "control-oracle"]);
const UNSET = "unset";

/**
 * CLI entry. Prints the audit through `log`; writes only with
 * `--drop-infra-errors --out P`. Throws on bad arguments.
 */
export function runResultsAudit(argv: string[], log: (line: string) => void = console.log): void {
  const arg = (name: string): string | undefined => {
    const i = argv.indexOf(name);
    return i >= 0 ? argv[i + 1] : undefined;
  };
  const bench = arg("--bench");
  if (bench !== "bfcl" && bench !== "sragents") {
    throw new Error(`--bench must be bfcl or sragents (got ${bench ?? "nothing"})`);
  }
  const agentArg = arg("--agent");
  if (!agentArg) throw new Error("--agent <path> is required");
  const agentPath = resolveRepoPath(agentArg);
  const drop = argv.includes("--drop-infra-errors");
  const outArg = arg("--out");
  const outPath = outArg ? resolveRepoPath(outArg) : undefined;
  if (drop !== (outPath !== undefined)) {
    throw new Error("--drop-infra-errors and --out <path> go together");
  }
  if (outPath && resolve(outPath) === resolve(agentPath)) {
    throw new Error("--out must differ from --agent (the input is never rewritten)");
  }
  // The old runner defaulted its cache source to the output file itself.
  const cachePaths = (arg("--cache")?.split(",").filter(Boolean) ?? [agentArg]).map(
    resolveRepoPath,
  );

  const agent = readRawJsonl(agentPath);
  const agentFile = { file: basename(agentPath), rows: agent.rows };
  const cacheFiles = cachePaths.map((p) =>
    resolve(p) === resolve(agentPath)
      ? agentFile
      : { file: basename(p), rows: readRawJsonl(p).rows },
  );
  const audited = [agentFile, ...cacheFiles.filter((f) => f !== agentFile)];

  for (const line of formatAudit({
    bench,
    agentPath,
    agentRows: agent.rows,
    malformed: agent.malformed,
    counts: auditRows(audited),
    stale: staleControlKeys(bench, cacheFiles),
    cacheFileCount: cacheFiles.length,
    mixed: mixedHarness(agent.rows),
    // SR's cell key has no SDK version, so only BFCL can split a cell on it.
    versionSplit: bench === "bfcl" ? versionSplitCells(agent.rows) : null,
  })) {
    log(line);
  }

  if (outPath) {
    const { drop, finalKept, newerKept } = planInfraDrop(bench, agent.rows);
    const kept = agent.lines.filter((l) => !l.row || !drop.has(l.row)).map((l) => l.text);
    writeFileSync(outPath, kept.length ? `${kept.join("\n")}\n` : "", "utf-8");
    log(
      `dropped ${drop.size} superseded infra-errored rows; kept ${finalKept} final ones ` +
        "(still counted in excluded_cells)" +
        (newerKept > 0
          ? ` and ${newerKept} superseded ones newer than their cell's kept row`
          : "") +
        `; wrote ${kept.length} lines to ${outPath}`,
    );
  }
}

/** Row counts per file × label × arm × model × error class × ratel_version, sorted. */
export function auditRows(files: AuditFile[]): AuditCount[] {
  const counts = new Map<string, AuditCount>();
  for (const { file, rows } of files) {
    for (const r of rows) {
      const count: AuditCount = {
        file,
        label: labelOf(r),
        arm: r.arm,
        model: r.model,
        error_class: errorClassOf(r) ?? "ok",
        ratel_version: r.ratel_version ?? UNSET,
        rows: 0,
        truncated: 0,
      };
      const key = sortKey(count);
      const entry = counts.get(key) ?? counts.set(key, count).get(key);
      if (!entry) continue;
      entry.rows++;
      if (isTruncated(r)) entry.truncated++;
    }
  }
  return [...counts.entries()].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)).map(([, c]) => c);
}

/**
 * Replay the OLD control cache over `files` (in order): per control key the
 * earliest `generated_at` wins, ties to the first row seen, errors included.
 * Returns the keys whose winner errored — the cells that rule re-served as errors.
 */
export function staleControlKeys(bench: Bench, files: AuditFile[]): StaleControl[] {
  const winners = new Map<string, { row: AuditRow; file: string }>();
  for (const { file, rows } of files) {
    for (const row of rows) {
      if (!CACHEABLE_ARMS.has(row.arm)) continue;
      // The BFCL runner skipped rows predating `ratel_version`.
      if (bench === "bfcl" && typeof row.ratel_version !== "string") continue;
      const key = controlKeyOf(bench, row);
      const prev = winners.get(key);
      if (!prev || (row.generated_at ?? "") < (prev.row.generated_at ?? "")) {
        winners.set(key, { row, file });
      }
    }
  }
  const out: StaleControl[] = [];
  for (const [key, { row, file }] of winners) {
    const cls = errorClassOf(row);
    if (cls === null) continue;
    out.push({
      key,
      arm: row.arm,
      model: row.model,
      label: labelOf(row),
      error_class: cls,
      error: row.error ?? "",
      file,
      generated_at: row.generated_at ?? "",
    });
  }
  return out.sort((a, b) => (a.key < b.key ? -1 : a.key > b.key ? 1 : 0));
}

/** Label × arm × model groups whose rows mix providers or requested output caps. */
export function mixedHarness(rows: AuditRow[]): MixedHarness[] {
  const groups = new Map<string, MixedHarness>();
  for (const r of rows) {
    const label = labelOf(r);
    const key = `${label}::${r.arm}::${r.model}`;
    const group =
      groups.get(key) ??
      groups.set(key, { label, arm: r.arm, model: r.model, providers: {}, caps: {} }).get(key);
    if (!group) continue;
    const provider = r.provider ?? UNSET;
    const cap = r.max_output_tokens == null ? UNSET : String(r.max_output_tokens);
    group.providers[provider] = (group.providers[provider] ?? 0) + 1;
    group.caps[cap] = (group.caps[cap] ?? 0) + 1;
  }
  return [...groups.values()].filter(
    (g) => Object.keys(g.providers).length > 1 || Object.keys(g.caps).length > 1,
  );
}

// ── helpers ──────────────────────────────────────────────────────────────────

interface RawLine {
  text: string;
  /** null when the line isn't valid JSON (kept verbatim by the drop). */
  row: AuditRow | null;
}

/** Every non-blank line with its parsed row; malformed lines are counted, not thrown. */
function readRawJsonl(path: string): { lines: RawLine[]; rows: AuditRow[]; malformed: number } {
  const lines: RawLine[] = [];
  for (const text of readFileSync(path, "utf-8").split("\n")) {
    if (!text.trim()) continue;
    let row: AuditRow | null = null;
    try {
      row = JSON.parse(text) as AuditRow;
    } catch {
      // kept as-is below
    }
    lines.push({ text, row });
  }
  const rows = lines.flatMap((l) => (l.row ? [l.row] : []));
  return { lines, rows, malformed: lines.length - rows.length };
}

function formatAudit(a: {
  bench: Bench;
  agentPath: string;
  agentRows: AuditRow[];
  malformed: number;
  counts: AuditCount[];
  stale: StaleControl[];
  cacheFileCount: number;
  mixed: MixedHarness[];
  /** null when the bench's cell key has no SDK version (sragents). */
  versionSplit: VersionSplitCell[] | null;
}): string[] {
  const errored = a.agentRows.filter((r) => r.error != null);
  const lines = [
    `results-audit (${a.bench}): ${a.agentPath}`,
    `errors: ${errored.length} of ${a.agentRows.length} rows (${tally(errored.map((r) => errorClassOf(r) ?? "ok"))})` +
      (a.malformed > 0 ? `; ${a.malformed} malformed lines` : ""),
    "",
    "errored/truncated rows by file · label · arm · model · class · ratel_version:",
  ];
  for (const c of a.counts) {
    if (c.error_class === "ok" && c.truncated === 0) continue;
    // Lead with the bad-row count: errored rows, or for ok groups the truncated ones.
    const count =
      c.error_class === "ok"
        ? `${c.truncated} truncated (of ${c.rows} ok)`
        : `${c.rows}${c.truncated > 0 ? ` (${c.truncated} truncated)` : ""}`;
    lines.push(
      `  ${c.file} · ${c.label} · ${c.arm} · ${c.model} · ${c.error_class} · ${c.ratel_version}: ${count}`,
    );
  }
  lines.push(
    "",
    `old-rule stale control keys: ${a.stale.length} (${tally(a.stale.map((s) => s.error_class))}); ` +
      `the earliest row per key across ${a.cacheFileCount} cache file(s) is an error`,
  );
  const byGroup = new Map<string, number>();
  for (const s of a.stale) {
    const key = `${s.model} · ${s.arm} · ${s.error_class}`;
    byGroup.set(key, (byGroup.get(key) ?? 0) + 1);
  }
  for (const [key, n] of [...byGroup].sort()) lines.push(`  ${key}: ${n}`);
  for (const s of a.stale) {
    lines.push(
      `    ${s.key} [${s.error_class}] ${s.file} @ ${s.generated_at}: ${s.error.slice(0, 80)}`,
    );
  }
  lines.push("", `mixed provider/cap groups: ${a.mixed.length}`);
  for (const m of a.mixed) {
    lines.push(
      `  ${m.label} · ${m.arm} · ${m.model}: providers {${tally(m.providers)}}; caps {${tally(m.caps)}}`,
    );
  }
  if (a.versionSplit) lines.push(...formatVersionChecks(a.versionSplit, errored));
  return lines;
}

/**
 * Label cells counted once per `ratel_version` (per label · arm · model), then the
 * `ratel_version`s of each label's infra-errored rows: what `--ratel-version` must be.
 */
function formatVersionChecks(split: VersionSplitCell[], errored: AuditRow[]): string[] {
  const lines = [
    "",
    `label cells under >1 ratel_version (each version counted as its own cell): ${split.length}`,
  ];
  const groups = new Map<string, { cells: number; versions: Record<string, number> }>();
  for (const c of split) {
    const key = `${c.label} · ${c.arm} · ${c.model}`;
    const group = groups.get(key) ?? groups.set(key, { cells: 0, versions: {} }).get(key);
    if (!group) continue;
    group.cells++;
    for (const [v, n] of Object.entries(c.versions))
      group.versions[v] = (group.versions[v] ?? 0) + n;
  }
  for (const [key, g] of [...groups].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))) {
    lines.push(`  ${key}: ${g.cells} cells (${tally(g.versions)})`);
  }
  lines.push(
    "",
    "infra-errored rows' ratel_version per label (re-drain with --ratel-version <that>):",
  );
  const byLabel = new Map<string, string[]>();
  for (const r of errored) {
    if (!isInfraError(r)) continue;
    const label = labelOf(r);
    (byLabel.get(label) ?? byLabel.set(label, []).get(label))?.push(r.ratel_version ?? UNSET);
  }
  for (const [label, versions] of [...byLabel].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))) {
    lines.push(`  ${label}: ${tally(versions)}`);
  }
  return lines;
}

/** `a 3, b 1` from a list of values or a precomputed count map. */
function tally(values: string[] | Record<string, number>): string {
  const counts: Record<string, number> = {};
  if (Array.isArray(values)) for (const v of values) counts[v] = (counts[v] ?? 0) + 1;
  else Object.assign(counts, values);
  return Object.entries(counts)
    .map(([k, n]) => `${k} ${n}`)
    .join(", ");
}

function labelOf(r: AuditRow): string {
  return r.ratel_ai_core_version ?? "unknown";
}

function sortKey(c: AuditCount): string {
  return [c.file, c.label, c.arm, c.model, c.error_class, c.ratel_version].join("\u0000");
}

/**
 * Which `transient|access` rows `--drop-infra-errors` removes: those superseded by
 * another row of the same cell (the summarizers' key and rule) and not newer than
 * it, so no summary metric, count or group timestamp changes. Also counts the
 * infra rows kept: final ones (the cell's winner) and newer superseded ones.
 */
function planInfraDrop(
  bench: Bench,
  rows: AuditRow[],
): { drop: Set<AuditRow>; finalKept: number; newerKept: number } {
  const keyOf = cellKeyOf(bench);
  const winners = new Map(supersede(rows, keyOf).map((r) => [keyOf(r), r]));
  const drop = new Set<AuditRow>();
  let finalKept = 0;
  let newerKept = 0;
  for (const r of rows) {
    if (!isInfraError(r)) continue;
    const winner = winners.get(keyOf(r));
    if (winner === r) finalKept++;
    else if ((r.generated_at ?? "") > (winner?.generated_at ?? "")) newerKept++;
    else drop.add(r);
  }
  return { drop, finalKept, newerKept };
}

/** A cell's identity within one label's history, as the bench's summarizer supersedes on. */
function cellKeyOf(bench: Bench): (r: AuditRow) => string {
  return bench === "bfcl"
    ? (r) => labelledCellKeyOf(r as unknown as CellResult)
    : (r) => sragentsCellKeyOf(r as unknown as SragentsSelectCell);
}

/**
 * Version-agnostic control identity, as each runner's cache keyed it before U2: BFCL
 * `runner.ts` controlKeyOf (`scenario::arm::model::run[::p<pool>]`), SR
 * `sragents-select.ts` controlKey (`scenario::arm::model::pool|null::run`). A
 * deliberate frozen copy: the audit replays the OLD cache rule.
 */
function controlKeyOf(bench: Bench, r: AuditRow): string {
  if (bench === "sragents") {
    return `${r.scenario_id}::${r.arm}::${r.model}::${r.pool_size ?? "null"}::${r.run_index}`;
  }
  const base = `${r.scenario_id}::${r.arm}::${r.model}::${r.run_index}`;
  return r.pool_size === null ? base : `${base}::p${r.pool_size}`;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  try {
    runResultsAudit(process.argv.slice(2));
  } catch (err) {
    console.error(`results-audit: ${(err as Error).message}`);
    process.exit(1);
  }
}
