import { appendFileSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { APICallError } from "ai";
import { afterEach, describe, expect, it } from "vitest";
import { createCampaignBudget, createMemoryBudgetStore } from "./campaign-budget.js";
import { newRetryStats, SpendJournalError, withRetry } from "./llm-retry.js";
import {
  inputTokenBound,
  openSpendLedger,
  reconcileCampaignBudget,
  spendRecorder,
} from "./spend-ledger.js";

const dirs: string[] = [];
afterEach(() => {
  for (const dir of dirs) rmSync(dir, { recursive: true, force: true });
  dirs.length = 0;
});

function path(): string {
  const dir = mkdtempSync(join(tmpdir(), "spend-ledger-"));
  dirs.push(dir);
  return join(dir, "attempts.jsonl");
}

describe("durable live spend", () => {
  it("reserves and claims before each physical retry, then settles usage", async () => {
    const file = path();
    const ledger = openSpendLedger(file);
    const store = createMemoryBudgetStore(0.0001);
    const price = { inputPer1M: 1, outputPer1M: 2, cachedInputPer1M: 1, cacheCreationPer1M: 1 };
    const budget = createCampaignBudget(store, {
      "bedrock/m": { price, maxInputTokens: 20, maxOutputTokens: 10 },
    });
    let calls = 0;
    const model = {
      provider: "test",
      modelId: "m",
      async doGenerate(_options?: unknown) {
        calls++;
        expect((await store.snapshot()).reservedTicks).toBeGreaterThan(0);
        if (calls === 1)
          throw new APICallError({
            message: "retry",
            url: "https://example.test",
            requestBodyValues: {},
            statusCode: 503,
          });
        return { usage: { inputTokens: 5, outputTokens: 2 } };
      },
    };
    const wrapped = withRetry(model as never, {
      policy: { maxAttempts: 2, baseMs: 1, maxDelayMs: 1, maxTotalWaitMs: 2 },
      stats: newRetryStats(),
      sleep: async () => {},
      attempt: spendRecorder(
        ledger,
        { kind: "bfcl", cellKey: "cell", model: "bedrock/m", price },
        budget,
      ),
    }) as typeof model;
    await wrapped.doGenerate({ maxOutputTokens: 10 } as never);
    expect(calls).toBe(2);
    // The 503 answered with an error before inference: free, and its reservation is returned.
    expect(ledger.summary()).toMatchObject({ attempts: 2, unknown: 0, completeness: "complete" });
    expect(await store.snapshot()).toMatchObject({
      spentTicks: 90000,
      reservedTicks: 0,
      unresolved: 0,
    });
  });

  it("replays a journaled response after budget settlement was unavailable", async () => {
    const file = path();
    const ledger = openSpendLedger(file);
    const store = createMemoryBudgetStore(0.0001);
    const price = { inputPer1M: 1, outputPer1M: 2, cachedInputPer1M: 1, cacheCreationPer1M: 1 };
    const route = { "bedrock/m": { price, maxInputTokens: 20, maxOutputTokens: 10 } };
    const failing = createCampaignBudget(
      {
        ...store,
        settle: async () => {
          throw new Error("store unavailable");
        },
      },
      route,
    );
    const attempt = spendRecorder(
      ledger,
      { kind: "bfcl", cellKey: "cell", model: "bedrock/m", price },
      failing,
    );
    const id = await attempt.start({ maxOutputTokens: 10 });
    await expect(
      attempt.finish(id, { usage: { inputTokens: 5, outputTokens: 2 } }, undefined, false),
    ).rejects.toThrow(/store unavailable/);
    expect(await store.snapshot()).toMatchObject({ spentTicks: 0, reservedTicks: 400000 });
    const resumed = openSpendLedger(file);
    await reconcileCampaignBudget(resumed, createCampaignBudget(store, route));
    await reconcileCampaignBudget(resumed, createCampaignBudget(store, route));
    expect(await store.snapshot()).toMatchObject({ spentTicks: 90000, reservedTicks: 0 });
  });

  it("fences duplicate worker delivery while a new logical retry gets a fresh reservation", async () => {
    const ledger = openSpendLedger(path());
    const store = createMemoryBudgetStore(0.0001);
    const price = { inputPer1M: 1, outputPer1M: 2, cachedInputPer1M: 1, cacheCreationPer1M: 1 };
    const budget = createCampaignBudget(store, {
      "bedrock/m": { price, maxInputTokens: 20, maxOutputTokens: 10 },
    });
    const context = {
      kind: "bfcl" as const,
      cellKey: "cell",
      model: "bedrock/m",
      price,
      attemptOrdinal: 1,
      runId: "first-delivery",
    };
    await spendRecorder(ledger, context, budget).start({ maxOutputTokens: 10 });
    await expect(
      spendRecorder(ledger, { ...context, runId: "duplicate-delivery" }, budget).start({
        maxOutputTokens: 10,
      }),
    ).rejects.toThrow(SpendJournalError);
    await spendRecorder(ledger, { ...context, attemptOrdinal: 2 }, budget).start({
      maxOutputTokens: 10,
    });
    expect(ledger.summary()).toMatchObject({ attempts: 2, unresolved: 2 });
    expect(await store.snapshot()).toMatchObject({ reservedTicks: 800000 });
  });

  it("advances a settled physical attempt when resuming before its output row", async () => {
    const file = path();
    const ledger = openSpendLedger(file);
    const store = { ...createMemoryBudgetStore(0.0001), campaignId: "campaign" };
    const price = { inputPer1M: 1, outputPer1M: 2, cachedInputPer1M: 1, cacheCreationPer1M: 1 };
    const route = { "bedrock/m": { price, maxInputTokens: 20, maxOutputTokens: 10 } };
    const budget = createCampaignBudget(store, route);
    const context = {
      kind: "bfcl" as const,
      cellKey: "cell",
      model: "bedrock/m",
      price,
      attemptOrdinal: 1,
    };
    const first = spendRecorder(ledger, context, budget);
    const firstId = await first.start({ maxOutputTokens: 10 });
    await first.finish(firstId, { usage: { inputTokens: 5, outputTokens: 2 } }, undefined, false);

    const resumed = openSpendLedger(file);
    await reconcileCampaignBudget(resumed, createCampaignBudget(store, route));
    const resumedId = await spendRecorder(resumed, context, budget).start({ maxOutputTokens: 10 });

    expect(resumedId).not.toBe(firstId);
    expect(resumed.summary()).toMatchObject({ attempts: 2, unresolved: 1 });
  });

  it("does not journal a second dispatched attempt from another worker", async () => {
    const firstLedger = openSpendLedger(path());
    const duplicateLedger = openSpendLedger(path());
    const store = createMemoryBudgetStore(0.0001);
    const price = { inputPer1M: 1, outputPer1M: 2, cachedInputPer1M: 1, cacheCreationPer1M: 1 };
    const budget = createCampaignBudget(store, {
      "bedrock/m": { price, maxInputTokens: 20, maxOutputTokens: 10 },
    });
    const context = { kind: "bfcl" as const, cellKey: "same-cell", model: "bedrock/m", price };
    await spendRecorder(firstLedger, context, budget).start({ maxOutputTokens: 10 });
    await expect(
      spendRecorder(duplicateLedger, context, budget).start({ maxOutputTokens: 10 }),
    ).rejects.toThrow(SpendJournalError);
    expect(firstLedger.summary().attempts).toBe(1);
    expect(duplicateLedger.summary().attempts).toBe(0);
  });

  it("never invokes the provider when the reservation service is unavailable", async () => {
    const ledger = openSpendLedger(path());
    const store = createMemoryBudgetStore(0.0001);
    const price = { inputPer1M: 1, outputPer1M: 2, cachedInputPer1M: 1, cacheCreationPer1M: 1 };
    const budget = createCampaignBudget(
      {
        ...store,
        reserve: async () => {
          throw new Error("unavailable");
        },
      },
      { "bedrock/m": { price, maxInputTokens: 20, maxOutputTokens: 10 } },
    );
    let calls = 0;
    const model = {
      provider: "test",
      modelId: "m",
      async doGenerate(_options?: unknown) {
        calls++;
        return {};
      },
    };
    const wrapped = withRetry(model as never, {
      policy: { maxAttempts: 2, baseMs: 1, maxDelayMs: 1, maxTotalWaitMs: 2 },
      stats: newRetryStats(),
      attempt: spendRecorder(
        ledger,
        { kind: "bfcl", cellKey: "cell", model: "bedrock/m", price },
        budget,
      ),
    }) as typeof model;
    await expect(wrapped.doGenerate({ maxOutputTokens: 10 } as never)).rejects.toThrow(
      /spend journal failed: unavailable/,
    );
    expect(calls).toBe(0);
    expect(ledger.summary().attempts).toBe(0);
  });
  it("isolates a current version from historical comparison attempts in one journal", () => {
    const file = path();
    const ledger = openSpendLedger(file);
    const price = {
      inputPer1M: 1,
      outputPer1M: 1,
      cachedInputPer1M: 1,
      cacheCreationPer1M: 1,
    };
    ledger.dispatch({
      id: "old",
      runId: "historical",
      kind: "bfcl",
      scope: "bfcl/old",
      cellKey: "old-cell",
      model: "bedrock/m",
      price,
    });
    ledger.settle("old", { status: "completed", usage: { inputTokens: 100, outputTokens: 0 } });
    ledger.dispatch({
      id: "current",
      runId: "current",
      kind: "bfcl",
      scope: "bfcl/new",
      cellKey: "new-cell",
      model: "bedrock/m",
      price,
    });
    ledger.settle("current", { status: "completed", usage: { inputTokens: 20, outputTokens: 0 } });
    expect(ledger.summary("bfcl/new")).toMatchObject({ attempts: 1, knownUsd: 0.00002 });
    expect(ledger.hasRun("historical", "bfcl/new")).toBe(false);
    expect(ledger.summary()).toMatchObject({ attempts: 2, knownUsd: 0.00012 });
    expect(openSpendLedger(file).summary("bfcl/new")).toMatchObject({
      attempts: 1,
      knownUsd: 0.00002,
    });
  });
  it("keeps a dispatched request unresolved across a crash and counts a retry separately", () => {
    const file = path();
    const ledger = openSpendLedger(file);
    ledger.dispatch({
      id: "first",
      kind: "bfcl",
      cellKey: "cell-1",
      model: "bedrock/m",
      price: null,
    });
    const resumed = openSpendLedger(file);
    expect(resumed.summary()).toMatchObject({
      attempts: 1,
      unresolved: 1,
      completeness: "partial",
    });
    resumed.dispatch({
      id: "retry",
      kind: "bfcl",
      cellKey: "cell-1",
      model: "bedrock/m",
      price: null,
    });
    resumed.settle("retry", { status: "failed", usage: null });
    expect(openSpendLedger(file).summary()).toMatchObject({ attempts: 2, unresolved: 1 });
  });

  it("keeps an unscoped legacy crash visible when its version cannot be trusted", () => {
    const file = path();
    openSpendLedger(file).dispatch({
      id: "phase10-crashed",
      kind: "bfcl",
      cellKey: "0.12.0::cell",
      model: "bedrock/m",
      price: null,
    });

    expect(openSpendLedger(file).summary("bfcl/0.12.0")).toMatchObject({
      attempts: 0,
      unresolved: 0,
      unattributedAttempts: 1,
      completeness: "partial",
    });
  });

  it("attributes an unscoped paid attempt from a canonical BFCL cell key", () => {
    const file = path();
    const ledger = openSpendLedger(file);
    ledger.dispatch({
      id: "phase10-paid",
      runId: "legacy-run",
      kind: "bfcl",
      cellKey: "0.12.0::scenario::control-baseline::bedrock/m::0::p1",
      model: "bedrock/m",
      price: { inputPer1M: 1, outputPer1M: 2, cachedInputPer1M: 1, cacheCreationPer1M: 1 },
    });
    ledger.settle("phase10-paid", {
      status: "completed",
      usage: { inputTokens: 10, outputTokens: 5 },
    });

    const resumed = openSpendLedger(file);
    expect(resumed.summary("bfcl/0.12.0")).toMatchObject({
      attempts: 1,
      unresolved: 0,
      knownUsd: 0.00002,
      completeness: "complete",
    });
    expect(resumed.hasRun("legacy-run", "bfcl/0.12.0")).toBe(true);
    expect(resumed.summary("bfcl/other")).toMatchObject({
      attempts: 0,
      knownUsd: 0,
      completeness: "complete",
    });
  });

  it("persists a response before the cell checkpoint and never double-counts replay", () => {
    const file = path();
    const ledger = openSpendLedger(file);
    const request = {
      id: "paid",
      kind: "bfcl" as const,
      cellKey: "cell-1",
      model: "bedrock/m",
      price: {
        inputPer1M: 2,
        outputPer1M: 4,
        cachedInputPer1M: 1,
        cacheCreationPer1M: 3,
      },
    };
    ledger.dispatch(request);
    const response = {
      status: "completed" as const,
      usage: {
        inputTokens: 100,
        outputTokens: 50,
        cachedInputTokens: 20,
        cacheCreationInputTokens: 10,
      },
    };
    ledger.settle("paid", response);
    const resumed = openSpendLedger(file);
    resumed.dispatch(request);
    resumed.settle("paid", response);
    expect(resumed.summary()).toMatchObject({
      attempts: 1,
      unresolved: 0,
      knownUsd: 0.00039,
      completeness: "complete",
    });
  });

  it("recovers a torn response write as unresolved spend before retrying", () => {
    const file = path();
    const ledger = openSpendLedger(file);
    ledger.dispatch({
      id: "response",
      kind: "bfcl",
      cellKey: "cell",
      model: "bedrock/m",
      price: null,
    });
    appendFileSync(file, '{"type":"settle","id":"response","value":');
    const resumed = openSpendLedger(file);
    expect(resumed.summary()).toMatchObject({
      attempts: 1,
      unresolved: 1,
      completeness: "partial",
    });
    resumed.dispatch({
      id: "retry",
      kind: "bfcl",
      cellKey: "cell",
      model: "bedrock/m",
      price: null,
    });
    expect(openSpendLedger(file).summary()).toMatchObject({ attempts: 2, unresolved: 2 });
  });

  it("keeps unknown usage and missing prices out of the known total", () => {
    const file = path();
    const ledger = openSpendLedger(file);
    ledger.dispatch({
      id: "judge",
      kind: "judge",
      cellKey: "cell-1",
      model: "bedrock/j",
      price: null,
    });
    ledger.settle("judge", { status: "completed", usage: { inputTokens: 10, outputTokens: 5 } });
    ledger.dispatch({
      id: "failed",
      kind: "bfcl",
      cellKey: "cell-2",
      model: "bedrock/m",
      price: null,
    });
    ledger.settle("failed", { status: "failed", usage: null });
    expect(ledger.summary()).toMatchObject({
      attempts: 2,
      unknown: 2,
      knownUsd: 0,
      completeness: "partial",
    });
  });

  it("keeps unpriced local inference out of paid unknowns", () => {
    const ledger = openSpendLedger(path());
    ledger.dispatch({
      id: "local",
      kind: "bfcl",
      cellKey: "cell",
      model: "ollama:qwen",
      price: null,
    });
    ledger.settle("local", { status: "completed", usage: null });
    expect(ledger.summary()).toMatchObject({
      attempts: 1,
      knownUsd: 0,
      unknown: 0,
      completeness: "complete",
    });
  });

  it("journals failed and successful physical calls independently", async () => {
    const file = path();
    const ledger = openSpendLedger(file);
    let calls = 0;
    const model = {
      specificationVersion: "v2" as const,
      provider: "test",
      modelId: "m",
      supportedUrls: {},
      async doGenerate(_options?: unknown) {
        calls++;
        if (calls === 1) {
          const error = Object.assign(
            new APICallError({
              message: "temporary",
              url: "https://example.test",
              requestBodyValues: {},
              statusCode: 503,
            }),
            { usage: { inputTokens: 5, outputTokens: 2 } },
          );
          throw error;
        }
        return { usage: { inputTokens: 10, outputTokens: 3 } };
      },
      async doStream(): Promise<never> {
        throw new Error("unused");
      },
    };
    const wrapped = withRetry(model as never, {
      policy: { maxAttempts: 2, baseMs: 1, maxDelayMs: 1, maxTotalWaitMs: 2 },
      stats: newRetryStats(),
      sleep: async () => {},
      attempt: spendRecorder(ledger, {
        kind: "bfcl",
        cellKey: "cell-1",
        model: "bedrock/m",
        price: { inputPer1M: 1, outputPer1M: 2, cachedInputPer1M: 1, cacheCreationPer1M: 1 },
      }),
    }) as typeof model;
    await wrapped.doGenerate({ prompt: [] });
    const settled = readFileSync(file, "utf8")
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line))
      .filter((event) => event.type === "settle");
    expect(settled[1].value.rawUsage).toMatchObject({ inputTokens: 10, outputTokens: 3 });
    expect(openSpendLedger(file).summary()).toMatchObject({
      attempts: 2,
      unresolved: 0,
      knownUsd: 0.000025,
      completeness: "complete",
    });
  });

  it("records an aborted call's returned usage without treating it as free", async () => {
    const file = path();
    const ledger = openSpendLedger(file);
    const controller = new AbortController();
    const model = {
      specificationVersion: "v2" as const,
      provider: "test",
      modelId: "m",
      supportedUrls: {},
      async doGenerate(_options?: unknown) {
        controller.abort();
        throw Object.assign(new Error("aborted"), { usage: { inputTokens: 20, outputTokens: 3 } });
      },
      async doStream(): Promise<never> {
        throw new Error("unused");
      },
    };
    const wrapped = withRetry(model as never, {
      policy: { maxAttempts: 1, baseMs: 1, maxDelayMs: 1, maxTotalWaitMs: 1 },
      stats: newRetryStats(),
      attempt: spendRecorder(ledger, {
        kind: "bfcl",
        cellKey: "cell",
        model: "bedrock/m",
        price: { inputPer1M: 1, outputPer1M: 2, cachedInputPer1M: 1, cacheCreationPer1M: 1 },
      }),
    }) as typeof model;
    await expect(wrapped.doGenerate({ abortSignal: controller.signal })).rejects.toThrow("aborted");
    const events = readFileSync(file, "utf8")
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line));
    expect(events[1].value).toMatchObject({
      status: "aborted",
      usage: { inputTokens: 20, outputTokens: 3 },
    });
    expect(ledger.summary()).toMatchObject({ knownUsd: 0.000026, completeness: "complete" });
  });

  it("keeps a call that got no provider answer unknown and reserved", async () => {
    const ledger = openSpendLedger(path());
    const store = createMemoryBudgetStore(0.0001);
    const price = { inputPer1M: 1, outputPer1M: 2, cachedInputPer1M: 1, cacheCreationPer1M: 1 };
    const budget = createCampaignBudget(store, {
      "bedrock/m": { price, maxInputTokens: 20, maxOutputTokens: 10 },
    });
    const attempt = spendRecorder(
      ledger,
      { kind: "bfcl", cellKey: "cell", model: "bedrock/m", price },
      budget,
    );
    const timedOut = await attempt.start({ maxOutputTokens: 10 });
    await attempt.finish(timedOut, undefined, new Error("socket hang up"), false);
    const gatewayTimeout = await attempt.start({ maxOutputTokens: 10 });
    const status504 = new APICallError({
      message: "gateway timeout",
      url: "https://example.test",
      requestBodyValues: {},
      statusCode: 504,
    });
    await attempt.finish(gatewayTimeout, undefined, status504, false);
    expect(ledger.summary()).toMatchObject({ unknown: 2, completeness: "partial" });
    expect(await store.snapshot()).toMatchObject({ reservedTicks: 800000, unresolved: 2 });
  });

  it("records a provider rejection as zero billable usage with its status", async () => {
    const file = path();
    const ledger = openSpendLedger(file);
    const attempt = spendRecorder(ledger, {
      kind: "sragents",
      cellKey: "cell",
      model: "bedrock/m",
      price: { inputPer1M: 1, outputPer1M: 2, cachedInputPer1M: 1, cacheCreationPer1M: 1 },
    });
    const id = await attempt.start({});
    const throttled = new APICallError({
      message: "throttled",
      url: "https://example.test",
      requestBodyValues: {},
      statusCode: 429,
    });
    await attempt.finish(id, undefined, throttled, false);
    const settle = readFileSync(file, "utf8")
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line))
      .find((event) => event.type === "settle");
    expect(settle.value).toMatchObject({
      status: "failed",
      usage: { inputTokens: 0, outputTokens: 0 },
      rejectedStatus: 429,
    });
  });

  it("sizes a campaign reservation from the prompt instead of the route's input bound", async () => {
    const ledger = openSpendLedger(path());
    const store = createMemoryBudgetStore(100);
    const price = { inputPer1M: 1, outputPer1M: 2, cachedInputPer1M: 1, cacheCreationPer1M: 1 };
    const budget = createCampaignBudget(store, {
      "bedrock/m": { price, maxInputTokens: 2_000_000, maxOutputTokens: 10 },
    });
    const options = {
      maxOutputTokens: 10,
      prompt: [{ role: "user", content: [{ type: "text", text: "hello" }] }],
    };
    const bound = inputTokenBound(options);
    expect(bound).toBeGreaterThan(JSON.stringify(options.prompt).length);
    expect(bound).toBeLessThan(10_000);
    const attempt = spendRecorder(
      ledger,
      { kind: "bfcl", cellKey: "cell", model: "bedrock/m", price },
      budget,
    );
    await attempt.start(options);
    expect((await store.snapshot()).reservedTicks).toBe((bound ?? 0) * 10_000 + 10 * 20_000);
  });

  it("bounds input only for text it can count", () => {
    const text = { type: "text", text: "é".repeat(100) };
    const textBound = inputTokenBound({ prompt: [{ role: "user", content: [text] }] }) ?? 0;
    // UTF-8 bytes, not UTF-16 code units, bound byte-level tokens.
    expect(textBound).toBeGreaterThan(200);
    const withTools = inputTokenBound({
      prompt: [{ role: "user", content: [text] }],
      tools: [{ type: "function", name: "t", inputSchema: { description: "x".repeat(500) } }],
    });
    expect(withTools).toBeGreaterThan(textBound + 500);
    expect(
      inputTokenBound({
        prompt: [{ role: "user", content: [{ type: "file", mediaType: "image/png", data: "u" }] }],
      }),
    ).toBeUndefined();
    expect(
      inputTokenBound({
        prompt: [
          {
            role: "tool",
            content: [{ type: "tool-result", output: { type: "content", value: [] } }],
          },
        ],
      }),
    ).toBeUndefined();
    expect(inputTokenBound(undefined)).toBeUndefined();
  });

  it("reconciles a resumed journal with one ledger listing and settles only the drift", async () => {
    const file = path();
    const ledger = openSpendLedger(file);
    const memory = createMemoryBudgetStore(0.001);
    let settles = 0;
    const store = {
      ...memory,
      settle: async (id: string, cost: number | null) => {
        settles++;
        await memory.settle(id, cost);
      },
    };
    const price = { inputPer1M: 1, outputPer1M: 2, cachedInputPer1M: 1, cacheCreationPer1M: 1 };
    const route = { "bedrock/m": { price, maxInputTokens: 20, maxOutputTokens: 10 } };
    const budget = createCampaignBudget(store, route);
    const context = { kind: "bfcl" as const, model: "bedrock/m", price };
    for (const cellKey of ["a", "b"]) {
      const attempt = spendRecorder(ledger, { ...context, cellKey }, budget);
      const id = await attempt.start({ maxOutputTokens: 10 });
      await attempt.finish(id, { usage: { inputTokens: 5, outputTokens: 2 } }, undefined, false);
    }
    const unsettled = spendRecorder(
      ledger,
      { ...context, cellKey: "c" },
      createCampaignBudget({ ...store, settle: async () => {} }, route),
    );
    const id = await unsettled.start({ maxOutputTokens: 10 });
    await unsettled.finish(id, { usage: { inputTokens: 5, outputTokens: 2 } }, undefined, false);
    settles = 0;

    await reconcileCampaignBudget(openSpendLedger(file), budget);

    expect(settles).toBe(1);
    expect(await memory.snapshot()).toMatchObject({ spentTicks: 270000, unresolved: 0 });
  });
});
