import { appendFileSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { APICallError } from "ai";
import { afterEach, describe, expect, it } from "vitest";
import { newRetryStats, withRetry } from "./llm-retry.js";
import { openSpendLedger, spendRecorder } from "./spend-ledger.js";

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
});
