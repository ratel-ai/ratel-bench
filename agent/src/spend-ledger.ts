import { createHash, randomUUID } from "node:crypto";
import {
  closeSync,
  existsSync,
  fsyncSync,
  mkdirSync,
  openSync,
  readFileSync,
  truncateSync,
  writeFileSync,
} from "node:fs";
import { dirname } from "node:path";
import {
  BudgetContentionError,
  BudgetLimitedError,
  type CampaignBudget,
} from "./campaign-budget.js";
import { type AttemptRecorder, SpendJournalError } from "./llm-retry.js";
import { type ModelPrice, xaiCostTicks } from "./metering.js";
import { ratedTicks } from "./money-ticks.js";

export interface SpendDispatch {
  id: string;
  /** Stable logical cell retry number; physical wrapper retries add their own ordinal. */
  attemptOrdinal?: number;
  workNamespace?: string;
  /** Budget identity; distinct campaigns can share a historical attempt journal. */
  campaignId?: string;
  /** Invocation whose output row can later be matched to this journal. */
  runId?: string;
  /** Benchmark/version whose live spend this attempt belongs to. */
  scope?: string;
  kind: "bfcl" | "sragents" | "judge";
  cellKey: string;
  model: string;
  servingProvider?: string;
  publisher?: string;
  resolvedModel?: string;
  vertexLocation?: string;
  adapterProvider?: string;
  adapterModel?: string;
  price: ModelPrice | null;
}

export interface SpendUsage {
  inputTokens?: number;
  outputTokens?: number;
  cachedInputTokens?: number;
  cacheCreationInputTokens?: number;
  totalTokens?: number;
}

export interface SpendSettlement {
  status: "completed" | "failed" | "aborted";
  usage: SpendUsage | null;
  /** Provider's original usage payload, retained for later repricing/audit. */
  rawUsage?: unknown;
  /** xAI billed amount, in 10^-10 USD ticks, when supplied by the provider. */
  providerCostTicks?: number;
}

export interface SpendSummary {
  attempts: number;
  unresolved: number;
  unknown: number;
  knownUsd: number;
  completeness: "complete" | "partial";
  /** Legacy live rows without evidence of provider-attempt journaling. */
  untrackedRows?: number;
  /** Legacy attempts compatible with this benchmark but lacking trustworthy scope evidence. */
  unattributedAttempts?: number;
}

interface Attempt {
  dispatch: SpendDispatch;
  settlement?: SpendSettlement;
}

interface Totals {
  attempts: number;
  unresolved: number;
  unknown: number;
  knownTicks: number;
}

type Event =
  | { type: "dispatch"; value: SpendDispatch }
  | { type: "settle"; id: string; value: SpendSettlement };

/** Open an append-only, fsynced provider-attempt journal. Dispatch is durable before I/O. */
export function openSpendLedger(path: string) {
  const attempts = new Map<string, Attempt>();
  const runIds = new Set<string>();
  const scopedRunIds = new Map<string, Set<string>>();
  const totals: Totals = { attempts: 0, unresolved: 0, unknown: 0, knownTicks: 0 };
  const scoped = new Map<string, Totals>();
  const forScope = (scope: string): Totals => {
    let value = scoped.get(scope);
    if (!value) {
      value = { attempts: 0, unresolved: 0, unknown: 0, knownTicks: 0 };
      scoped.set(scope, value);
    }
    return value;
  };
  if (existsSync(path)) {
    const contents = readFileSync(path, "utf8");
    const end = contents.lastIndexOf("\n") + 1;
    // A torn final write is discarded; its preceding dispatch remains unresolved.
    if (end < contents.length) truncateSync(path, Buffer.byteLength(contents.slice(0, end)));
    for (const line of contents.slice(0, end).split("\n")) {
      if (line) apply(JSON.parse(line) as Event);
    }
  }

  function apply(event: Event): void {
    if (event.type === "dispatch") {
      const old = attempts.get(event.value.id);
      if (old && JSON.stringify(old.dispatch) !== JSON.stringify(event.value))
        throw new Error(`conflicting spend attempt ${event.value.id}`);
      if (!old) {
        attempts.set(event.value.id, { dispatch: event.value });
        for (const current of [
          totals,
          ...(event.value.scope ? [forScope(event.value.scope)] : []),
        ]) {
          current.attempts++;
          current.unresolved++;
        }
      }
      if (event.value.runId) {
        runIds.add(event.value.runId);
        if (event.value.scope) {
          const scopedIds = scopedRunIds.get(event.value.scope) ?? new Set<string>();
          scopedIds.add(event.value.runId);
          scopedRunIds.set(event.value.scope, scopedIds);
        }
      }
    } else {
      const attempt = attempts.get(event.id);
      if (!attempt) throw new Error(`settlement without dispatch: ${event.id}`);
      if (attempt.settlement && JSON.stringify(attempt.settlement) !== JSON.stringify(event.value))
        throw new Error(`conflicting spend settlement ${event.id}`);
      if (!attempt.settlement) {
        attempt.settlement = event.value;
        const ticks = costTicks(attempt.dispatch.model, attempt.dispatch.price, event.value);
        for (const current of [
          totals,
          ...(attempt.dispatch.scope ? [forScope(attempt.dispatch.scope)] : []),
        ]) {
          current.unresolved--;
          if (ticks === null) current.unknown++;
          else current.knownTicks += ticks;
        }
      }
    }
  }

  function append(event: Event): void {
    mkdirSync(dirname(path), { recursive: true });
    const fd = openSync(path, "a");
    try {
      writeFileSync(fd, `${JSON.stringify(event)}\n`);
      fsyncSync(fd);
    } finally {
      closeSync(fd);
    }
    apply(event);
  }

  return {
    entries(): ReadonlyArray<Readonly<Attempt>> {
      return [...attempts.values()];
    },
    hasRun(runId: string, scope?: string): boolean {
      return scope === undefined
        ? runIds.has(runId)
        : (scopedRunIds.get(scope)?.has(runId) ?? false) ||
            [...attempts.values()].some(
              ({ dispatch }) =>
                !dispatch.scope && dispatch.runId === runId && legacyScopeOf(dispatch) === scope,
            );
    },
    dispatch(value: SpendDispatch): void {
      if (attempts.has(value.id)) {
        apply({ type: "dispatch", value });
        return;
      }
      append({ type: "dispatch", value });
    },
    settle(id: string, value: SpendSettlement): void {
      const attempt = attempts.get(id);
      if (!attempt) throw new Error(`settlement without dispatch: ${id}`);
      const old = attempt.settlement;
      if (old) {
        apply({ type: "settle", id, value });
        return;
      }
      append({ type: "settle", id, value });
    },
    summary(scope?: string): SpendSummary {
      const value = scope === undefined ? totals : { ...(scoped.get(scope) ?? emptyTotals()) };
      let unattributedAttempts = 0;
      if (scope !== undefined) {
        for (const attempt of attempts.values()) {
          if (attempt.dispatch.scope) continue;
          const inferred = legacyScopeOf(attempt.dispatch);
          if (inferred === scope) addAttempt(value, attempt);
          else if (!inferred && benchmarkMatchesScope(attempt.dispatch.kind, scope)) {
            unattributedAttempts++;
          }
        }
      }
      return {
        attempts: value.attempts,
        unresolved: value.unresolved,
        unknown: value.unknown,
        knownUsd: value.knownTicks / 10_000_000_000,
        completeness:
          value.unresolved || value.unknown || unattributedAttempts ? "partial" : "complete",
        ...(unattributedAttempts ? { unattributedAttempts } : {}),
      };
    },
  };
}

function emptyTotals(): Totals {
  return { attempts: 0, unresolved: 0, unknown: 0, knownTicks: 0 };
}

function addAttempt(totals: Totals, attempt: Attempt): void {
  totals.attempts++;
  if (!attempt.settlement) {
    totals.unresolved++;
    return;
  }
  const ticks = costTicks(attempt.dispatch.model, attempt.dispatch.price, attempt.settlement);
  if (ticks === null) totals.unknown++;
  else totals.knownTicks += ticks;
}

function legacyScopeOf(dispatch: SpendDispatch): string | undefined {
  if (dispatch.kind !== "bfcl" && dispatch.kind !== "judge") return undefined;
  const parts = dispatch.cellKey.split("::");
  if (parts.length !== 5 && parts.length !== 6) return undefined;
  const [version, scenario, arm, model, runIndex, pool] = parts;
  if (!version || !scenario || !arm || !model || !/^(0|[1-9]\d*)$/.test(runIndex)) return undefined;
  if (pool !== undefined && !/^p(0|[1-9]\d*)$/.test(pool)) return undefined;
  return `bfcl/${version}`;
}

function benchmarkMatchesScope(kind: SpendDispatch["kind"], scope: string): boolean {
  if (scope.startsWith("bfcl/")) return kind === "bfcl" || kind === "judge";
  if (scope.startsWith("sragents/")) return kind === "sragents";
  return false;
}

export type SpendLedger = ReturnType<typeof openSpendLedger>;

/** Replay budget settlement after a crash between journal fsync and store acknowledgement. */
export async function reconcileCampaignBudget(
  ledger: SpendLedger,
  budget: CampaignBudget,
): Promise<void> {
  for (const { dispatch, settlement } of ledger.entries()) {
    if (dispatch.campaignId !== budget.campaignId || !settlement) continue;
    await budget.settle(dispatch.id, costTicks(dispatch.model, dispatch.price, settlement));
  }
}

/** A distinct ID for each physical provider dispatch, including every retry. */
export function newSpendAttemptId(): string {
  return randomUUID();
}

/** Attach the journal to a model's physical doGenerate calls. */
export function spendRecorder(
  ledger: SpendLedger,
  context: Omit<SpendDispatch, "id">,
  budget?: CampaignBudget,
): AttemptRecorder {
  let physicalOrdinal = 0;
  return {
    async start(options) {
      physicalOrdinal++;
      const id = budget
        ? createHash("sha256")
            .update(
              JSON.stringify([
                budget.campaignId,
                context.workNamespace ?? "evaluation",
                context.kind,
                context.scope,
                context.cellKey,
                context.model,
                context.attemptOrdinal ?? 1,
                physicalOrdinal,
              ]),
            )
            .digest("hex")
        : newSpendAttemptId();
      if (budget) {
        const maxOutputTokens = (options as { maxOutputTokens?: number } | undefined)
          ?.maxOutputTokens;
        try {
          await budget.reserve(id, context.model, maxOutputTokens, context.price);
        } catch (error) {
          if (error instanceof BudgetLimitedError || error instanceof BudgetContentionError)
            throw error;
          throw new SpendJournalError(error);
        }
        try {
          await budget.claim(id);
        } catch (error) {
          throw new SpendJournalError(error);
        }
      }
      try {
        ledger.dispatch({ ...context, id, ...(budget ? { campaignId: budget.campaignId } : {}) });
      } catch (error) {
        // The claim may already authorize a concurrent worker. Retain its
        // reservation; no request can follow a failed journal write here.
        throw new SpendJournalError(error);
      }
      return id;
    },
    async finish(id, result, error, aborted) {
      try {
        const response = error === undefined ? result : error;
        const object =
          response && typeof response === "object" ? (response as Record<string, unknown>) : {};
        const rawUsage =
          object.usage ?? (object.response as { usage?: unknown } | undefined)?.usage;
        const usage = spendUsageOf(rawUsage);
        const billedTicks = xaiCostTicks(object.providerMetadata, object.usage);
        const settlement: SpendSettlement = {
          status: aborted ? "aborted" : error === undefined ? "completed" : "failed",
          usage,
          rawUsage,
          ...(billedTicks === undefined ? {} : { providerCostTicks: billedTicks }),
        };
        ledger.settle(id, settlement);
        if (budget) await budget.settle(id, costTicks(context.model, context.price, settlement));
      } catch (failure) {
        throw new SpendJournalError(failure);
      }
    },
  };
}

function spendUsageOf(raw: unknown): SpendUsage | null {
  if (!raw || typeof raw !== "object") return null;
  const usage = raw as Record<string, unknown>;
  const input = tokenCount(usage.inputTokens);
  const output = tokenCount(usage.outputTokens);
  if (input === undefined || output === undefined) return null;
  const inputDetail =
    usage.inputTokens && typeof usage.inputTokens === "object"
      ? (usage.inputTokens as Record<string, unknown>)
      : {};
  return {
    inputTokens: input,
    outputTokens: output,
    cachedInputTokens:
      tokenCount(inputDetail.cacheRead) ??
      tokenCount(usage.cachedInputTokens) ??
      tokenCount((usage.inputTokenDetails as Record<string, unknown> | undefined)?.cacheReadTokens),
    cacheCreationInputTokens:
      tokenCount(inputDetail.cacheWrite) ??
      tokenCount(usage.cacheCreationInputTokens) ??
      tokenCount(
        (usage.inputTokenDetails as Record<string, unknown> | undefined)?.cacheWriteTokens,
      ),
    totalTokens: tokenCount(usage.totalTokens),
  };
}

function tokenCount(value: unknown): number | undefined {
  if (validCount(value)) return value;
  if (value && typeof value === "object") {
    const total = (value as { total?: unknown }).total;
    if (validCount(total)) return total;
  }
  return undefined;
}

function costTicks(
  model: string,
  price: ModelPrice | null,
  settlement: SpendSettlement,
): number | null {
  if (model.startsWith("ollama:")) return 0;
  if (settlement.providerCostTicks !== undefined) {
    return Number.isSafeInteger(settlement.providerCostTicks) && settlement.providerCostTicks >= 0
      ? settlement.providerCostTicks
      : null;
  }
  const usage = settlement.usage;
  if (!price || !usage || !validCount(usage.inputTokens) || !validCount(usage.outputTokens))
    return null;
  const read = usage.cachedInputTokens ?? 0;
  const write = usage.cacheCreationInputTokens ?? 0;
  if (!validCount(read) || !validCount(write) || read + write > usage.inputTokens) return null;
  const uncached = usage.inputTokens - read - write;
  const parts = [
    ratedTicks(uncached, price.inputPer1M),
    ratedTicks(usage.outputTokens, price.outputPer1M),
    ratedTicks(read, price.cachedInputPer1M),
    ratedTicks(write, price.cacheCreationPer1M),
  ];
  if (parts.some((part) => part === null)) return null;
  const total = parts.reduce<bigint>((sum, part) => sum + (part ?? 0n), 0n);
  return total <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(total) : null;
}

function validCount(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
}
