import { randomUUID } from "node:crypto";
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
import { type AttemptRecorder, SpendJournalError } from "./llm-retry.js";
import { type ModelPrice, xaiCostTicks } from "./metering.js";

export interface SpendDispatch {
  id: string;
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
    hasRun(runId: string, scope?: string): boolean {
      return scope === undefined
        ? runIds.has(runId)
        : (scopedRunIds.get(scope)?.has(runId) ?? false);
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
      const value =
        scope === undefined
          ? totals
          : (scoped.get(scope) ?? {
              attempts: 0,
              unresolved: 0,
              unknown: 0,
              knownTicks: 0,
            });
      return {
        attempts: value.attempts,
        unresolved: value.unresolved,
        unknown: value.unknown,
        knownUsd: value.knownTicks / 10_000_000_000,
        completeness: value.unresolved || value.unknown ? "partial" : "complete",
      };
    },
  };
}

export type SpendLedger = ReturnType<typeof openSpendLedger>;

/** A distinct ID for each physical provider dispatch, including every retry. */
export function newSpendAttemptId(): string {
  return randomUUID();
}

/** Attach the journal to a model's physical doGenerate calls. */
export function spendRecorder(
  ledger: SpendLedger,
  context: Omit<SpendDispatch, "id">,
): AttemptRecorder {
  return {
    start() {
      const id = newSpendAttemptId();
      try {
        ledger.dispatch({ ...context, id });
      } catch (error) {
        throw new SpendJournalError(error);
      }
      return id;
    },
    finish(id, result, error, aborted) {
      try {
        const response = error === undefined ? result : error;
        const object =
          response && typeof response === "object" ? (response as Record<string, unknown>) : {};
        const rawUsage =
          object.usage ?? (object.response as { usage?: unknown } | undefined)?.usage;
        const usage = spendUsageOf(rawUsage);
        const billedTicks = xaiCostTicks(object.providerMetadata, object.usage);
        ledger.settle(id, {
          status: aborted ? "aborted" : error === undefined ? "completed" : "failed",
          usage,
          rawUsage,
          ...(billedTicks === undefined ? {} : { providerCostTicks: billedTicks }),
        });
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

/** Decimal catalog rates become exact rational ticks, rounded up per category. */
function ratedTicks(tokens: number, rate: number): bigint | null {
  if (!validCount(tokens) || !Number.isFinite(rate) || rate < 0) return null;
  const match = /^(\d+)(?:\.(\d+))?(?:e([+-]?\d+))?$/i.exec(rate.toString());
  if (!match) return null;
  const digits = BigInt(`${match[1]}${match[2] ?? ""}`);
  const scale = (match[2]?.length ?? 0) - Number(match[3] ?? 0);
  const numerator = BigInt(tokens) * digits * 10_000n * (scale < 0 ? 10n ** BigInt(-scale) : 1n);
  const denominator = scale > 0 ? 10n ** BigInt(scale) : 1n;
  return (numerator + denominator - 1n) / denominator;
}

function validCount(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
}
