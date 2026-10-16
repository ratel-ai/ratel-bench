import { randomUUID } from "node:crypto";
import type { ModelPrice } from "./metering.js";
import { campaignCeilingTicks, ratedTicks } from "./money-ticks.js";

export interface BudgetRoute {
  price: ModelPrice | null;
  /** Provider-enforced maximum billable input, including cache writes. */
  maxInputTokens: number;
  /** Provider-enforced maximum billable output, including reasoning tokens. */
  maxOutputTokens: number;
}

export interface BudgetRouteRequest {
  model: string;
  price: ModelPrice | null;
  requestedOutputTokens: number | undefined;
}

export interface BudgetSnapshot {
  ceilingTicks: number;
  spentTicks: number;
  reservedTicks: number;
  remainingTicks: number;
  unresolved: number;
  discrepant: boolean;
}

/** One durable attempt as the store records it. */
export interface BudgetAttempt {
  attemptId: string;
  amountTicks: number;
  state: string;
  costTicks?: number;
}

/** Every operation is atomic, durable, and idempotent by attempt ID at the store boundary. */
export interface BudgetStore {
  readonly campaignId: string;
  reserve(id: string, amountTicks: number): Promise<"granted" | "wait" | "exhausted">;
  claim(id: string): Promise<boolean>;
  settle(id: string, costTicks: number | null): Promise<void>;
  release(id: string): Promise<void>;
  snapshot(): Promise<BudgetSnapshot>;
  /** Every attempt of the campaign in one listing, when the store supports it. */
  attempts?(): Promise<readonly BudgetAttempt[]>;
}

export class BudgetLimitedError extends Error {
  readonly status = "budget_limited";
  constructor(readonly snapshot: BudgetSnapshot) {
    super("campaign budget exhausted");
  }
}

export class BudgetContentionError extends Error {
  constructor() {
    super("campaign budget held by unresolved attempts; retry after drain");
  }
}

export interface CampaignBudget {
  readonly campaignId: string;
  preflight(requests: readonly BudgetRouteRequest[]): void;
  /**
   * Reserve the call's worst-case cost. `inputTokens` is a proven upper bound on the
   * call's billable input; the route's `maxInputTokens` caps it and stands in when absent.
   */
  reserve(
    id: string,
    model: string,
    requestedOutputTokens: number | undefined,
    price?: ModelPrice | null,
    inputTokens?: number,
  ): Promise<void>;
  claim(id: string): Promise<void>;
  admit(id: string, model: string, requestedOutputTokens: number | undefined): Promise<void>;
  settle(id: string, costTicks: number | null): Promise<void>;
  release(id: string): Promise<void>;
  snapshot(): Promise<BudgetSnapshot>;
  attempts?(): Promise<readonly BudgetAttempt[]>;
}

/** Validate every selected paid route before starting any model work. */
export function createCampaignBudget(
  store: BudgetStore,
  routes: Record<string, BudgetRoute>,
  options: {
    waitForChange?: () => Promise<void>;
    isDrained?: () => Promise<boolean>;
    maxWaitMs?: number;
  } = {},
): CampaignBudget {
  const amountOf = (route: BudgetRoute, inputTokens: number): bigint | null => {
    if (!route.price) return null;
    const inputRate = Math.max(
      route.price.inputPer1M,
      route.price.cachedInputPer1M,
      route.price.cacheCreationPer1M,
    );
    const input = ratedTicks(inputTokens, inputRate);
    const output = ratedTicks(route.maxOutputTokens, route.price.outputPer1M);
    return input === null || output === null ? null : input + output;
  };
  for (const [model, route] of Object.entries(routes)) {
    if (
      !route.price ||
      Object.values(route.price).some((rate) => !Number.isFinite(rate) || rate < 0)
    )
      throw new Error(`campaign preflight: missing or invalid price for ${model}`);
    if (!positiveCount(route.maxInputTokens))
      throw new Error(`campaign preflight: unsupported input bound for ${model}`);
    if (!positiveCount(route.maxOutputTokens))
      throw new Error(`campaign preflight: unsupported output/reasoning bound for ${model}`);
    const amount = amountOf(route, route.maxInputTokens);
    if (amount === null || amount > BigInt(Number.MAX_SAFE_INTEGER))
      throw new Error(`campaign preflight: unsupported usage bound for ${model}`);
  }
  const waitForChange =
    options.waitForChange ?? (() => new Promise<void>((resolve) => setTimeout(resolve, 100)));
  const maxWaitMs = options.maxWaitMs ?? 30_000;
  const budget: CampaignBudget = {
    campaignId: store.campaignId,
    preflight(requests) {
      for (const request of requests) {
        const route = routes[request.model];
        if (!route) throw new Error(`campaign preflight: unpriced route ${request.model}`);
        if (!samePrice(route.price, request.price))
          throw new Error(`campaign preflight: price snapshot mismatch for ${request.model}`);
        if (
          !positiveCount(request.requestedOutputTokens) ||
          request.requestedOutputTokens > route.maxOutputTokens
        )
          throw new Error(`campaign preflight: unsupported output bound for ${request.model}`);
      }
    },
    async reserve(id, model, requestedOutputTokens, price, inputTokens) {
      const route = routes[model];
      if (!route) throw new Error(`campaign preflight: unpriced route ${model}`);
      if (inputTokens !== undefined && !positiveCount(inputTokens))
        throw new Error(`campaign: invalid input bound for ${model}`);
      // Preflight proved the route bound's amount is a safe integer; a smaller input is too.
      const amount = Number(
        amountOf(route, Math.min(inputTokens ?? route.maxInputTokens, route.maxInputTokens)),
      );
      if (price !== undefined && !samePrice(route.price, price))
        throw new Error(`campaign preflight: price snapshot mismatch for ${model}`);
      if (!positiveCount(requestedOutputTokens) || requestedOutputTokens > route.maxOutputTokens)
        throw new Error(`campaign preflight: unsupported output bound for ${model}`);
      const started = Date.now();
      for (;;) {
        const decision = await store.reserve(id, amount);
        if (decision === "exhausted") throw new BudgetLimitedError(await store.snapshot());
        if (decision === "granted") return;
        if (await options.isDrained?.()) throw new BudgetLimitedError(await store.snapshot());
        if (Date.now() - started >= maxWaitMs) throw new BudgetContentionError();
        await waitForChange();
      }
    },
    async claim(id) {
      if (!(await store.claim(id))) throw new Error(`attempt ${id} already dispatched`);
    },
    async admit(id, model, requestedOutputTokens) {
      await budget.reserve(id, model, requestedOutputTokens);
      await budget.claim(id);
    },
    settle: (id, costTicks) => store.settle(id, costTicks),
    release: (id) => store.release(id),
    snapshot: () => store.snapshot(),
    ...(store.attempts ? { attempts: store.attempts.bind(store) } : {}),
  };
  return budget;
}

/** Offline store fixture; production persistence is supplied by the caller. */
export function createMemoryBudgetStore(
  ceilingUsd: number,
  campaignId = randomUUID(),
): BudgetStore {
  const ceilingTicks = campaignCeilingTicks(ceilingUsd);
  if (ceilingTicks === null)
    throw new Error("campaign ceiling must be positive, finite, and representable");
  const entries = new Map<
    string,
    {
      amount: number;
      state: "reserved" | "claimed" | "settled" | "unknown" | "released";
      cost?: number;
    }
  >();
  let spentTicks = 0;
  let discrepant = false;
  const snapshot = (): BudgetSnapshot => {
    let reservedTicks = 0;
    let unresolved = 0;
    for (const entry of entries.values()) {
      if (["reserved", "claimed", "unknown"].includes(entry.state)) {
        reservedTicks += entry.amount;
        unresolved++;
      }
    }
    return {
      ceilingTicks,
      spentTicks,
      reservedTicks,
      remainingTicks: Math.max(0, ceilingTicks - spentTicks - reservedTicks),
      unresolved,
      discrepant,
    };
  };
  return {
    campaignId,
    async reserve(id, amountTicks) {
      if (!Number.isSafeInteger(amountTicks) || amountTicks < 0)
        throw new Error("invalid reservation");
      if (discrepant) throw new Error("campaign usage exceeded reserved bound");
      const existing = entries.get(id);
      if (existing) {
        if (existing.amount !== amountTicks || existing.state === "released")
          throw new Error(`conflicting reservation ${id}`);
        return "granted";
      }
      const current = snapshot();
      if (current.spentTicks + amountTicks > ceilingTicks) return "exhausted";
      if (current.spentTicks + current.reservedTicks + amountTicks > ceilingTicks) return "wait";
      entries.set(id, { amount: amountTicks, state: "reserved" });
      return "granted";
    },
    async claim(id) {
      const entry = entries.get(id);
      if (!entry) throw new Error(`reservation ${id} absent`);
      if (entry.state !== "reserved") return false;
      entry.state = "claimed";
      return true;
    },
    async settle(id, costTicks) {
      const entry = entries.get(id);
      if (!entry || !["claimed", "settled", "unknown"].includes(entry.state))
        throw new Error(`attempt ${id} was not dispatched`);
      if (costTicks === null) {
        if (entry.state === "claimed") entry.state = "unknown";
        return;
      }
      if (!Number.isSafeInteger(costTicks) || costTicks < 0) throw new Error("invalid settlement");
      if (entry.state === "settled") {
        if (entry.cost !== costTicks) throw new Error(`conflicting settlement ${id}`);
        return;
      }
      entry.cost = costTicks;
      entry.state = "settled";
      spentTicks += costTicks;
      if (costTicks > entry.amount) discrepant = true;
    },
    async release(id) {
      const entry = entries.get(id);
      if (!entry) return;
      if (entry.state === "reserved") entry.state = "released";
      else if (entry.state !== "released")
        throw new Error(`attempt ${id} may have been dispatched`);
    },
    async snapshot() {
      return snapshot();
    },
    async attempts() {
      return [...entries].map(([attemptId, entry]) => ({
        attemptId,
        amountTicks: entry.amount,
        state: entry.state,
        ...(entry.cost === undefined ? {} : { costTicks: entry.cost }),
      }));
    },
  };
}

function positiveCount(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value > 0;
}

function samePrice(a: ModelPrice | null, b: ModelPrice | null): boolean {
  return (
    a !== null &&
    b !== null &&
    a.inputPer1M === b.inputPer1M &&
    a.outputPer1M === b.outputPer1M &&
    a.cachedInputPer1M === b.cachedInputPer1M &&
    a.cacheCreationPer1M === b.cacheCreationPer1M
  );
}
