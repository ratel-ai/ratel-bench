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
  /** HTTP status of a provider error answer that billed no inference; `usage` is then zero. */
  rejectedStatus?: number;
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

interface SpendGeneration {
  baseScope: string;
  scope: string;
}

interface Totals {
  attempts: number;
  unresolved: number;
  unknown: number;
  knownTicks: number;
}

/**
 * Tokens a provider may add around the serialized request: chat-template markers,
 * a tool-use preamble, per-message framing. Generous on purpose: an input bound
 * below the billed input marks the campaign discrepant and stops admission.
 */
const PROVIDER_PROMPT_OVERHEAD_TOKENS = 4096;
/** Prompt parts whose tokens the request's UTF-8 bytes bound (each token is ≥ 1 byte). */
const COUNTABLE_PARTS = new Set(["text", "reasoning", "tool-call", "tool-result"]);
/** Settled-attempt replays in flight at once during a resume. */
const RECONCILE_CONCURRENCY = 16;

type Event =
  | { type: "generation"; value: SpendGeneration }
  | { type: "dispatch"; value: SpendDispatch }
  | { type: "settle"; id: string; value: SpendSettlement };

/** Open an append-only, fsynced provider-attempt journal. Dispatch is durable before I/O. */
export function openSpendLedger(path: string) {
  const attempts = new Map<string, Attempt>();
  const runIds = new Set<string>();
  const scopedRunIds = new Map<string, Set<string>>();
  const currentScopes = new Map<string, string>();
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
    if (event.type === "generation") {
      currentScopes.set(event.value.baseScope, event.value.scope);
    } else if (event.type === "dispatch") {
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
    currentScope(baseScope: string): string {
      return currentScopes.get(baseScope) ?? baseScope;
    },
    startGeneration(baseScope: string): string {
      const scope = `${baseScope}/generation/${randomUUID()}`;
      append({ type: "generation", value: { baseScope, scope } });
      return scope;
    },
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

/**
 * Replay budget settlement after a crash between journal fsync and store acknowledgement.
 * One store listing finds the attempts already settled; only the drift is written.
 */
export async function reconcileCampaignBudget(
  ledger: SpendLedger,
  budget: CampaignBudget,
): Promise<void> {
  const listed = budget.attempts
    ? new Map((await budget.attempts()).map((attempt) => [attempt.attemptId, attempt]))
    : undefined;
  const pending: Array<() => Promise<void>> = [];
  for (const { dispatch, settlement } of ledger.entries()) {
    if (dispatch.campaignId !== budget.campaignId || !settlement) continue;
    const cost = costTicks(dispatch.model, dispatch.price, settlement);
    const current = listed?.get(dispatch.id);
    if (current?.state === "settled" && current.costTicks === cost) continue;
    if (current?.state === "unknown" && cost === null) continue;
    pending.push(() => budget.settle(dispatch.id, cost));
  }
  const worker = async (): Promise<void> => {
    for (let next = pending.shift(); next; next = pending.shift()) await next();
  };
  await Promise.all(Array.from({ length: RECONCILE_CONCURRENCY }, worker));
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
      const inputTokens = budget ? inputTokenBound(options) : undefined;
      let id: string;
      do {
        physicalOrdinal++;
        id = budget
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
                  // A resumed call with a different prompt is a different reservation.
                  ...(inputTokens === undefined ? [] : [inputTokens]),
                ]),
              )
              .digest("hex")
          : newSpendAttemptId();
      } while (budget && isSettledAttempt(ledger, id));
      if (budget) {
        const maxOutputTokens = (options as { maxOutputTokens?: number } | undefined)
          ?.maxOutputTokens;
        try {
          await budget.reserve(id, context.model, maxOutputTokens, context.price, inputTokens);
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
        const reported = spendUsageOf(rawUsage);
        const rejectedStatus =
          reported === null && error !== undefined && !aborted
            ? rejectedStatusOf(error)
            : undefined;
        const usage = rejectedStatus === undefined ? reported : { inputTokens: 0, outputTokens: 0 };
        const billedTicks = xaiCostTicks(object.providerMetadata, object.usage);
        const settlement: SpendSettlement = {
          status: aborted ? "aborted" : error === undefined ? "completed" : "failed",
          usage,
          rawUsage,
          ...(billedTicks === undefined ? {} : { providerCostTicks: billedTicks }),
          ...(rejectedStatus === undefined ? {} : { rejectedStatus }),
        };
        ledger.settle(id, settlement);
        if (budget) await budget.settle(id, costTicks(context.model, context.price, settlement));
      } catch (failure) {
        throw new SpendJournalError(failure);
      }
    },
  };
}

/**
 * A proven upper bound on a call's billable input tokens: the UTF-8 bytes of what
 * the request serializes, plus the provider overhead. Undefined when the prompt
 * holds media (or anything else bytes do not bound): the route bound then applies.
 */
export function inputTokenBound(options: unknown): number | undefined {
  if (!options || typeof options !== "object") return undefined;
  const { prompt, tools, toolChoice, responseFormat, providerOptions } = options as Record<
    string,
    unknown
  >;
  if (!Array.isArray(prompt) || !prompt.every(countableMessage)) return undefined;
  let serialized: string;
  try {
    serialized = JSON.stringify({ prompt, tools, toolChoice, responseFormat, providerOptions });
  } catch {
    return undefined;
  }
  return Buffer.byteLength(serialized, "utf8") + PROVIDER_PROMPT_OVERHEAD_TOKENS;
}

function countableMessage(message: unknown): boolean {
  const content = (message as { content?: unknown } | null)?.content;
  if (typeof content === "string") return true;
  return (
    Array.isArray(content) &&
    content.every((part: { type?: unknown; output?: { type?: unknown } } | null) => {
      if (!part || !COUNTABLE_PARTS.has(part.type as string)) return false;
      return part.type !== "tool-result" || part.output?.type !== "content";
    })
  );
}

/**
 * The status of a provider error answer that billed no inference: any 4xx/5xx
 * except the timeouts (408, 504), after which the provider may still have run
 * the call. Without such an answer (network error, abort) the cost is unknown.
 */
function rejectedStatusOf(error: unknown): number | undefined {
  const status = (error as { statusCode?: unknown } | null)?.statusCode;
  if (typeof status !== "number" || status < 400 || status > 599) return undefined;
  return status === 408 || status === 504 ? undefined : status;
}

function isSettledAttempt(ledger: SpendLedger, id: string): boolean {
  return ledger.entries().some((attempt) => attempt.dispatch.id === id && attempt.settlement);
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
