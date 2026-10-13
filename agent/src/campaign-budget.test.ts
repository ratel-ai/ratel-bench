import { describe, expect, it } from "vitest";
import { createCampaignBudget, createMemoryBudgetStore } from "./campaign-budget.js";

const price = { inputPer1M: 1, outputPer1M: 2, cachedInputPer1M: 0.5, cacheCreationPer1M: 3 };
const route = { price, maxInputTokens: 100, maxOutputTokens: 50 };

describe("campaign admission", () => {
  it("atomically reserves, claims once, and settles the observed amount", async () => {
    const store = createMemoryBudgetStore(0.0005);
    const budget = createCampaignBudget(store, { "bedrock/m": route });
    await budget.admit("one", "bedrock/m", 50);
    expect(await store.snapshot()).toMatchObject({ spentTicks: 0, reservedTicks: 4000000 });
    await expect(budget.admit("one", "bedrock/m", 50)).rejects.toThrow(/already dispatched/);
    await budget.settle("one", 100000);
    await budget.settle("one", 100000);
    expect(await store.snapshot()).toMatchObject({
      spentTicks: 100000,
      reservedTicks: 0,
      remainingTicks: 4900000,
    });
  });

  it("waits for temporary reservations then admits a concurrent attempt", async () => {
    const store = createMemoryBudgetStore(0.0005);
    let drain!: () => void;
    const budget = createCampaignBudget(
      store,
      { "bedrock/m": route },
      {
        waitForChange: () =>
          new Promise<void>((resolve) => {
            drain = resolve;
          }),
      },
    );
    await budget.admit("one", "bedrock/m", 50);
    const second = budget.admit("two", "bedrock/m", 50);
    await Promise.resolve();
    await budget.settle("one", 100000);
    drain();
    await second;
    expect(await store.snapshot()).toMatchObject({ spentTicks: 100000, reservedTicks: 4000000 });
  });

  it("reports confirmed exhaustion after settlement", async () => {
    const store = createMemoryBudgetStore(0.0004);
    const budget = createCampaignBudget(store, { "bedrock/m": route });
    await budget.admit("one", "bedrock/m", 50);
    await budget.settle("one", 4000000);
    await expect(budget.admit("two", "bedrock/m", 50)).rejects.toMatchObject({
      status: "budget_limited",
    });
  });

  it("limits coverage after drain when an ambiguous dispatch still holds headroom", async () => {
    const store = createMemoryBudgetStore(0.0005);
    await createCampaignBudget(store, { "bedrock/m": route }).admit("crashed", "bedrock/m", 50);
    const resumed = createCampaignBudget(
      store,
      { "bedrock/m": route },
      {
        isDrained: async () => true,
      },
    );
    await expect(resumed.admit("next", "bedrock/m", 50)).rejects.toMatchObject({
      status: "budget_limited",
    });
    expect(await store.snapshot()).toMatchObject({ reservedTicks: 4000000 });
  });

  it("retains ambiguous dispatched reservations across resume and fails closed on outage", async () => {
    const store = createMemoryBudgetStore(0.0005);
    await createCampaignBudget(store, { "bedrock/m": route }).admit("crashed", "bedrock/m", 50);
    const resumed = createCampaignBudget(store, { "bedrock/m": route });
    expect(await store.snapshot()).toMatchObject({ spentTicks: 0, reservedTicks: 4000000 });
    await expect(resumed.admit("crashed", "bedrock/m", 50)).rejects.toThrow(/already dispatched/);
    const unavailable = createCampaignBudget(
      {
        ...store,
        reserve: async () => {
          throw new Error("down");
        },
      },
      { "bedrock/m": route },
    );
    await expect(unavailable.admit("new", "bedrock/m", 50)).rejects.toThrow(/down/);
  });

  it("keeps missing usage reserved and stops admission after an observed bound violation", async () => {
    const store = createMemoryBudgetStore(0.001);
    const budget = createCampaignBudget(store, { "bedrock/m": route });
    await budget.admit("missing", "bedrock/m", 50);
    await budget.settle("missing", null);
    await budget.settle("missing", null);
    expect(await store.snapshot()).toMatchObject({
      spentTicks: 0,
      reservedTicks: 4000000,
      unresolved: 1,
    });
    await budget.admit("over", "bedrock/m", 50);
    await budget.settle("over", 4000001);
    expect(await store.snapshot()).toMatchObject({ discrepant: true });
    await expect(budget.admit("next", "bedrock/m", 50)).rejects.toThrow(/exceeded reserved bound/);
  });

  it("rejects missing prices, unsupported bounds, and output requests above the frozen cap", async () => {
    const store = createMemoryBudgetStore(1);
    expect(() => createCampaignBudget(store, { "bedrock/m": { ...route, price: null } })).toThrow(
      /price/,
    );
    expect(() =>
      createCampaignBudget(store, { "bedrock/m": { ...route, maxInputTokens: 0 } }),
    ).toThrow(/input/);
    const budget = createCampaignBudget(store, { "bedrock/m": route });
    await expect(budget.admit("bad", "bedrock/m", 51)).rejects.toThrow(/output/);
    expect(await store.snapshot()).toMatchObject({ reservedTicks: 0 });
    expect(() =>
      budget.preflight([{ model: "bedrock/m", price: null, requestedOutputTokens: 50 }]),
    ).toThrow(/price snapshot/);
    expect(() =>
      budget.preflight([{ model: "bedrock/m", price, requestedOutputTokens: undefined }]),
    ).toThrow(/output/);
  });
});
