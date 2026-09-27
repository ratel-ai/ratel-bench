import { APICallError, generateText } from "ai";
import { MockLanguageModelV3 } from "ai/test";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  CellTimeoutError,
  FatalProviderError,
  PROVIDER_ERROR_MARKER,
  RetriesExhaustedError,
} from "./cell-errors.js";
import {
  breakerLine,
  breakerThresholdFromEnv,
  cellRetry,
  createBreaker,
  DEFAULT_RETRY_SETTINGS,
  newRetryStats,
  PausableDeadline,
  type RetryPolicy,
  retryDelayMs,
  retryPolicyLabel,
  retrySettingsFromEnv,
  retrySettingsLine,
  sleep,
  withRetry,
} from "./llm-retry.js";

const POLICY: RetryPolicy = {
  maxAttempts: 4,
  baseMs: 1000,
  maxDelayMs: 8000,
  maxTotalWaitMs: 60_000,
};

function apiError(
  statusCode: number,
  headers?: Record<string, string>,
  message = `HTTP ${statusCode}`,
): APICallError {
  return new APICallError({
    message,
    url: "https://api.test/v1",
    requestBodyValues: {},
    statusCode,
    responseHeaders: headers,
  });
}

const V3_OK = {
  content: [{ type: "text" as const, text: "ok" }],
  finishReason: { unified: "stop" as const, raw: "end_turn" },
  usage: {
    inputTokens: { total: 1, noCache: 1, cacheRead: 0, cacheWrite: 0 },
    outputTokens: { total: 1, text: 1, reasoning: 0 },
  },
  warnings: [],
};

/** A model whose first `failures.length` calls throw those errors, then succeeds. */
function failingV3(failures: unknown[]): MockLanguageModelV3 {
  let call = 0;
  return new MockLanguageModelV3({
    provider: "mock-v3",
    modelId: "m3",
    doGenerate: async () => {
      const err = failures[call++];
      if (err !== undefined) throw err;
      return V3_OK;
    },
  });
}

/** A hand-rolled V2 model (the anthropic/bedrock spec) with the same failure script. */
function failingV2(failures: unknown[]) {
  let call = 0;
  return {
    specificationVersion: "v2" as const,
    provider: "mock-v2",
    modelId: "m2",
    supportedUrls: {},
    calls: 0,
    async doGenerate() {
      this.calls++;
      const err = failures[call++];
      if (err !== undefined) throw err;
      return {
        content: [{ type: "text" as const, text: "ok" }],
        finishReason: "stop" as const,
        usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
        warnings: [],
      };
    },
    async doStream(): Promise<never> {
      throw new Error("unused");
    },
  };
}

/** One direct `doGenerate` call (no SDK in between), as a real Promise. */
function generate(
  model: { doGenerate: MockLanguageModelV3["doGenerate"] },
  abortSignal?: AbortSignal,
): Promise<unknown> {
  return Promise.resolve(model.doGenerate({ prompt: [], abortSignal }));
}

function recordingSleep() {
  const waits: number[] = [];
  const fn = async (ms: number) => {
    waits.push(ms);
  };
  return { waits, fn };
}

describe("retryDelayMs", () => {
  it("equal jitter over a capped exponential: rng=0 → d/2, rng→1 → d", () => {
    expect(retryDelayMs(1, POLICY, () => 0)).toBe(500);
    expect(retryDelayMs(1, POLICY, () => 0.999999)).toBeCloseTo(1000, 0);
    expect(retryDelayMs(3, POLICY, () => 0)).toBe(2000);
    expect(retryDelayMs(3, POLICY, () => 0.5)).toBe(3000);
    // d = min(cap, base·2^(n−1)) = min(8000, 16000)
    expect(retryDelayMs(5, POLICY, () => 0)).toBe(4000);
    expect(retryDelayMs(5, POLICY, () => 0.999999)).toBeCloseTo(8000, 0);
  });
});

describe("withRetry", () => {
  it("retries transient errors and counts retries / throttled / wait (V3)", async () => {
    const model = failingV3([apiError(429), apiError(500), apiError(503)]);
    const stats = newRetryStats();
    const { waits, fn } = recordingSleep();
    const wrapped = withRetry(model, { policy: POLICY, stats, sleep: fn, random: () => 0 });

    const result = await generateText({ model: wrapped, prompt: "hi", maxRetries: 0 });

    expect(result.text).toBe("ok");
    expect(model.doGenerateCalls).toHaveLength(4);
    expect(waits).toEqual([500, 1000, 2000]);
    expect(stats).toEqual({ retries: 3, throttledRetries: 2, waitMs: 3500, fatal: false });
  });

  it("preserves specificationVersion/provider/modelId for a V2 and a V3 model", async () => {
    const v2 = failingV2([apiError(529)]);
    const v3 = failingV3([]);
    const stats = newRetryStats();
    const w2 = withRetry(v2, { policy: POLICY, stats, sleep: async () => {} });
    const w3 = withRetry(v3, { policy: POLICY, stats: newRetryStats(), sleep: async () => {} });

    expect([w2.specificationVersion, w2.provider, w2.modelId]).toEqual(["v2", "mock-v2", "m2"]);
    expect([w3.specificationVersion, w3.provider, w3.modelId]).toEqual(["v3", "mock-v3", "m3"]);

    const result = await generateText({ model: w2, prompt: "hi", maxRetries: 0 });
    expect(result.text).toBe("ok");
    expect(v2.calls).toBe(2);
    expect(stats.throttledRetries).toBe(1);
  });

  it("generateText(maxRetries:0) calls doGenerate exactly maxAttempts times, then RetriesExhaustedError", async () => {
    const model = failingV3(Array.from({ length: 10 }, () => apiError(503)));
    const stats = newRetryStats();
    const wrapped = withRetry(model, { policy: POLICY, stats, sleep: async () => {} });

    const err = await generateText({ model: wrapped, prompt: "hi", maxRetries: 0 }).catch(
      (e: unknown) => e,
    );

    expect(model.doGenerateCalls).toHaveLength(POLICY.maxAttempts);
    expect(err).toBeInstanceOf(RetriesExhaustedError);
    expect(APICallError.isInstance(err)).toBe(false);
    expect((err as RetriesExhaustedError).attempts).toBe(4);
    expect((err as RetriesExhaustedError).cause).toBeInstanceOf(APICallError);
    expect((err as Error).message).toBe("Failed after 4 attempts. Last error: HTTP 503");
    expect(stats.retries).toBe(3);
  });

  it("a fatal-marked 429 → exactly 1 call, FatalProviderError, stats.fatal", async () => {
    // A message no rule matches: only the marker makes this retryable 429 fatal.
    const gated = Object.assign(apiError(429, undefined, "ThrottlingException: Rate exceeded"), {
      [PROVIDER_ERROR_MARKER]: { kind: "fatal", scope: "model", reason: "gated" },
    });
    const model = failingV3([gated, gated]);
    const stats = newRetryStats();
    const wrapped = withRetry(model, { policy: POLICY, stats, sleep: async () => {} });

    const err = await generateText({ model: wrapped, prompt: "hi", maxRetries: 0 }).catch(
      (e: unknown) => e,
    );

    expect(model.doGenerateCalls).toHaveLength(1);
    expect(err).toBeInstanceOf(FatalProviderError);
    expect(APICallError.isInstance(err)).toBe(false);
    expect((err as Error).message).toBe("ThrottlingException: Rate exceeded");
    expect((err as Error).cause).toBe(gated);
    expect(stats).toMatchObject({ retries: 0, fatal: true });
  });

  it("an unmarked daily-cap 429 is fatal too: exactly 1 call", async () => {
    const model = failingV3([
      apiError(429, undefined, "Too many tokens per day, please wait before trying again."),
    ]);
    const stats = newRetryStats();
    const wrapped = withRetry(model, { policy: POLICY, stats, sleep: async () => {} });

    const err = await generate(wrapped).catch((e: unknown) => e);

    expect(err).toBeInstanceOf(FatalProviderError);
    expect(model.doGenerateCalls).toHaveLength(1);
    expect(stats).toMatchObject({ retries: 0, fatal: true });
  });

  it("an OpenAI insufficient_quota 429 is fatal: exactly 1 call", async () => {
    const quota = new APICallError({
      message: "You exceeded your current quota, please check your plan and billing details.",
      url: "https://api.test/v1",
      requestBodyValues: {},
      statusCode: 429,
      isRetryable: true,
      data: { error: { type: "insufficient_quota", code: "insufficient_quota" } },
    });
    const model = failingV3([quota, quota]);
    const stats = newRetryStats();
    const wrapped = withRetry(model, { policy: POLICY, stats, sleep: async () => {} });

    const err = await generateText({ model: wrapped, prompt: "hi", maxRetries: 0 }).catch(
      (e: unknown) => e,
    );

    expect(model.doGenerateCalls).toHaveLength(1);
    expect(err).toBeInstanceOf(FatalProviderError);
    expect(stats).toMatchObject({ retries: 0, fatal: true });
  });

  it("an access error (403) is fatal too", async () => {
    const model = failingV3([apiError(403)]);
    const stats = newRetryStats();
    const wrapped = withRetry(model, { policy: POLICY, stats, sleep: async () => {} });

    const err = await generate(wrapped).catch((e: unknown) => e as Error);

    expect(err).toBeInstanceOf(FatalProviderError);
    expect(stats.fatal).toBe(true);
  });

  it("rethrows every other error unchanged, without retrying", async () => {
    const badRequest = apiError(400);
    const boom = new Error("boom");
    for (const original of [badRequest, boom]) {
      const model = failingV3([original]);
      const wrapped = withRetry(model, {
        policy: POLICY,
        stats: newRetryStats(),
        sleep: async () => {},
      });
      const err = await generate(wrapped).catch((e: unknown) => e);
      expect(err).toBe(original);
      expect(model.doGenerateCalls).toHaveLength(1);
    }
  });

  it("honours Retry-After / retry-after-ms up to the cap: max(jittered, header)", async () => {
    const model = failingV3([
      apiError(429, { "retry-after": "5" }),
      apiError(429, { "retry-after-ms": "100" }),
    ]);
    const { waits, fn } = recordingSleep();
    const wrapped = withRetry(model, {
      policy: POLICY,
      stats: newRetryStats(),
      sleep: fn,
      random: () => 0,
    });

    await generate(wrapped);

    // 5s header beats the 500ms jitter; the 100ms header loses to the 1000ms jitter.
    expect(waits).toEqual([5000, 1000]);
  });

  it("retry-after-ms wins over retry-after", async () => {
    const model = failingV3([apiError(429, { "retry-after-ms": "3000", "retry-after": "7" })]);
    const { waits, fn } = recordingSleep();
    const wrapped = withRetry(model, {
      policy: POLICY,
      stats: newRetryStats(),
      sleep: fn,
      random: () => 0,
    });

    await generate(wrapped);

    expect(waits).toEqual([3000]);
  });

  it("reads an HTTP-date Retry-After against the clock (a past date counts as 0)", async () => {
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(new Date("2026-01-01T00:00:00Z"));
    try {
      const model = failingV3([
        apiError(429, { "retry-after": "Thu, 01 Jan 2026 00:00:05 GMT" }),
        apiError(429, { "retry-after": "Wed, 31 Dec 2025 23:59:00 GMT" }),
      ]);
      const { waits, fn } = recordingSleep();
      const wrapped = withRetry(model, {
        policy: POLICY,
        stats: newRetryStats(),
        sleep: fn,
        random: () => 0,
      });

      await generate(wrapped);

      expect(waits).toEqual([5000, 1000]);

      const late = failingV3([apiError(429, { "retry-after": "Thu, 01 Jan 2026 00:00:09 GMT" })]);
      const lateWaits = recordingSleep();
      const err = await generate(
        withRetry(late, { policy: POLICY, stats: newRetryStats(), sleep: lateWaits.fn }),
      ).catch((e: unknown) => e);
      expect(err).toBeInstanceOf(RetriesExhaustedError);
      expect((err as Error).message).toMatch(/\(Retry-After 9000ms > max delay 8000ms\)/);
      expect(late.doGenerateCalls).toHaveLength(1);
      expect(lateWaits.waits).toEqual([]);
    } finally {
      vi.useRealTimers();
    }
  });

  it("fails fast (transient) when Retry-After exceeds the cap or the remaining budget", async () => {
    for (const [header, policy] of [
      ["9", POLICY], // 9s > maxDelayMs 8s
      ["5", { ...POLICY, maxTotalWaitMs: 4000 }], // 5s > 4s budget
    ] as const) {
      const model = failingV3([apiError(429, { "retry-after": header })]);
      const { waits, fn } = recordingSleep();
      const stats = newRetryStats();
      const wrapped = withRetry(model, { policy, stats, sleep: fn });

      const err = await generate(wrapped).catch((e: unknown) => e);

      expect(err).toBeInstanceOf(RetriesExhaustedError);
      expect((err as Error).message).toMatch(/^Failed after 1 attempts \(Retry-After/);
      expect(model.doGenerateCalls).toHaveLength(1);
      expect(waits).toEqual([]);
      expect(stats.retries).toBe(0);
    }
  });

  it("stops at maxTotalWaitMs", async () => {
    const model = failingV3(Array.from({ length: 10 }, () => apiError(503)));
    const { waits, fn } = recordingSleep();
    const stats = newRetryStats();
    const policy = { ...POLICY, maxAttempts: 10, maxTotalWaitMs: 2000 };
    const wrapped = withRetry(model, { policy, stats, sleep: fn, random: () => 0 });

    const err = await generate(wrapped).catch((e: unknown) => e);

    // 500 + 1000 = 1500 waited; the next 2000 would pass the 2000 budget.
    expect(waits).toEqual([500, 1000]);
    expect(stats.waitMs).toBe(1500);
    expect(err).toBeInstanceOf(RetriesExhaustedError);
    expect((err as Error).message).toMatch(
      /^Failed after 3 attempts \(retry wait budget 2000ms: next wait 2000ms > 500ms left\)\. /,
    );
  });

  it("the wait budget is per stats, i.e. per cell: it spans the cell's calls", async () => {
    const model = failingV3([apiError(503), apiError(503), undefined, apiError(503)]);
    const { waits, fn } = recordingSleep();
    const stats = newRetryStats();
    const policy = { ...POLICY, maxAttempts: 10, maxTotalWaitMs: 1500 };
    const wrapped = withRetry(model, { policy, stats, sleep: fn, random: () => 0 });

    await generate(wrapped); // waits 500 + 1000: the whole budget
    const err = await generate(wrapped).catch((e: unknown) => e);

    expect(err).toBeInstanceOf(RetriesExhaustedError);
    expect((err as Error).message).toMatch(
      /^Failed after 1 attempts \(retry wait budget 1500ms: next wait 500ms > 0ms left\)\. /,
    );
    expect(waits).toEqual([500, 1000]);
    expect(stats).toMatchObject({ retries: 2, waitMs: 1500 });
    expect(model.doGenerateCalls).toHaveLength(4);
  });

  it("does not retry once the call's abort signal fired", async () => {
    const ac = new AbortController();
    const e503 = apiError(503);
    const model = new MockLanguageModelV3({
      doGenerate: async () => {
        ac.abort(new CellTimeoutError(10));
        throw e503;
      },
    });
    const stats = newRetryStats();
    const wrapped = withRetry(model, { policy: POLICY, stats, sleep: async () => {} });

    const err = await generate(wrapped, ac.signal).catch((e: unknown) => e);

    // The provider's own (transient) error, not a phantom retry.
    expect(err).toBe(e503);
    expect(model.doGenerateCalls).toHaveLength(1);
    expect(stats).toMatchObject({ retries: 0, throttledRetries: 0, waitMs: 0 });
  });

  it("sleeps on the call's own abort signal", async () => {
    const ac = new AbortController();
    const seen: (AbortSignal | undefined)[] = [];
    const wrapped = withRetry(failingV3([apiError(503)]), {
      policy: POLICY,
      stats: newRetryStats(),
      sleep: async (_ms, signal) => {
        seen.push(signal);
      },
    });

    await generate(wrapped, ac.signal);

    expect(seen).toHaveLength(1);
    expect(seen[0]).toBe(ac.signal);
  });

  it("an abort mid-backoff ends the call at once, without another attempt", async () => {
    vi.useFakeTimers();
    try {
      const ac = new AbortController();
      const reason = new CellTimeoutError(10);
      const model = failingV3(Array.from({ length: 10 }, () => apiError(503)));
      const stats = newRetryStats();
      const wrapped = withRetry(model, { policy: POLICY, stats, random: () => 0 });
      let settled: unknown;
      void generate(wrapped, ac.signal).catch((e: unknown) => {
        settled = e;
      });

      await vi.advanceTimersByTimeAsync(100); // mid the 500ms backoff
      ac.abort(reason);
      await vi.advanceTimersByTimeAsync(0);

      expect(settled).toBe(reason);
      expect(model.doGenerateCalls).toHaveLength(1);
      expect(stats.waitMs).toBe(0);
    } finally {
      vi.useRealTimers();
    }
  });

  it("pauses the deadline around each retry sleep", async () => {
    const model = failingV3([apiError(503)]);
    const events: string[] = [];
    const deadline = {
      pause: () => events.push("pause"),
      resume: () => events.push("resume"),
    };
    const wrapped = withRetry(model, {
      policy: POLICY,
      stats: newRetryStats(),
      deadline,
      sleep: async () => {
        events.push("sleep");
      },
    });

    await generate(wrapped);

    expect(events).toEqual(["pause", "sleep", "resume"]);
  });

  it("throws a clear error for a plain string model id", () => {
    expect(() => withRetry("gpt-5.4-mini", { policy: POLICY, stats: newRetryStats() })).toThrow(
      /model instance/,
    );
  });
});

describe("sleep", () => {
  it("resolves after ms and rejects with the abort reason when the signal fires", async () => {
    await expect(sleep(1)).resolves.toBeUndefined();
    const ac = new AbortController();
    const reason = new CellTimeoutError(5);
    const pending = sleep(10_000, ac.signal);
    ac.abort(reason);
    await expect(pending).rejects.toBe(reason);
    const already = new AbortController();
    already.abort(reason);
    await expect(sleep(10_000, already.signal)).rejects.toBe(reason);
  });
});

describe("PausableDeadline", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("fires after the unpaused ms with a CellTimeoutError reason", () => {
    vi.useFakeTimers();
    const d = new PausableDeadline(1000, { graceMs: 500 });
    vi.advanceTimersByTime(999);
    expect(d.signal.aborted).toBe(false);
    vi.advanceTimersByTime(1);
    expect(d.signal.aborted).toBe(true);
    expect(d.signal.reason).toBeInstanceOf(CellTimeoutError);
    expect((d.signal.reason as Error).message).toBe("run timed out after 1000ms");
    d.dispose();
  });

  it("pause() stops the clock; resume() restarts it with the remaining time", () => {
    vi.useFakeTimers();
    const d = new PausableDeadline(1000, { graceMs: 500 });
    vi.advanceTimersByTime(600);
    d.pause();
    vi.advanceTimersByTime(5000);
    expect(d.signal.aborted).toBe(false);
    d.resume();
    vi.advanceTimersByTime(399);
    expect(d.signal.aborted).toBe(false);
    vi.advanceTimersByTime(1);
    expect(d.signal.aborted).toBe(true);
    d.dispose();
  });

  it("race() rejects with a CellTimeoutError at deadline + grace when the work ignores the abort", async () => {
    vi.useFakeTimers();
    const d = new PausableDeadline(1000, { graceMs: 500 });
    const raced = d.race(new Promise<never>(() => {})).catch((e: unknown) => e);
    let settled = false;
    void raced.then(() => {
      settled = true;
    });
    await vi.advanceTimersByTimeAsync(1499);
    expect(d.signal.aborted).toBe(true);
    expect(settled).toBe(false);
    await vi.advanceTimersByTimeAsync(1);
    expect(settled).toBe(true);
    const err = await raced;
    expect(err).toBeInstanceOf(CellTimeoutError);
    expect((err as Error).message).toBe("run timed out after 1000ms");
    d.dispose();
  });

  it("race() passes the work's own result through", async () => {
    const d = new PausableDeadline(1000, { graceMs: 500 });
    await expect(d.race(Promise.resolve(7))).resolves.toBe(7);
    d.dispose();
  });

  it("dispose() clears the timers", () => {
    vi.useFakeTimers();
    const d = new PausableDeadline(1000, { graceMs: 500 });
    d.dispose();
    expect(vi.getTimerCount()).toBe(0);
    vi.advanceTimersByTime(5000);
    expect(d.signal.aborted).toBe(false);
  });

  it("dispose() after expiry clears the armed grace timer", () => {
    vi.useFakeTimers();
    const d = new PausableDeadline(1000, { graceMs: 500 });
    vi.advanceTimersByTime(1000);
    expect(d.signal.aborted).toBe(true);
    expect(vi.getTimerCount()).toBe(1);
    d.dispose();
    expect(vi.getTimerCount()).toBe(0);
  });

  it("measures active time on a monotonic clock: a wall-clock step doesn't move it", () => {
    vi.useFakeTimers();
    for (const step of [60_000, -60_000]) {
      const d = new PausableDeadline(1000, { graceMs: 500 });
      vi.advanceTimersByTime(100);
      vi.setSystemTime(Date.now() + step);
      d.pause();
      d.resume();
      vi.advanceTimersByTime(899);
      expect(d.signal.aborted).toBe(false);
      vi.advanceTimersByTime(1);
      expect(d.signal.aborted).toBe(true);
      d.dispose();
    }
  });
});

describe("cellRetry", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("run() releases the deadline's timers when the work resolves or rejects", async () => {
    vi.useFakeTimers();
    const ok = cellRetry(failingV3([]), 1000, DEFAULT_RETRY_SETTINGS);
    expect(vi.getTimerCount()).toBe(1);
    await expect(ok.run(async () => 1)).resolves.toBe(1);
    expect(vi.getTimerCount()).toBe(0);

    const boom = cellRetry(failingV3([]), 1000, DEFAULT_RETRY_SETTINGS);
    await expect(
      boom.run(async () => {
        throw new Error("boom");
      }),
    ).rejects.toThrow("boom");
    expect(vi.getTimerCount()).toBe(0);
  });

  it("run() releases the grace timer when the work settles after the deadline", async () => {
    vi.useFakeTimers();
    const r = cellRetry(failingV3([]), 1000, DEFAULT_RETRY_SETTINGS);
    const p = r.run(() => new Promise((resolve) => setTimeout(() => resolve(1), 1500)));
    await vi.advanceTimersByTimeAsync(1500);
    await expect(p).resolves.toBe(1);
    expect(r.signal.aborted).toBe(true);
    expect(vi.getTimerCount()).toBe(0);
  });

  it("logs one line per retry, tagged with the cell label", async () => {
    const lines: string[] = [];
    const r = cellRetry(
      failingV3([apiError(429, { "retry-after": "3" }), apiError(500)]),
      60_000,
      {
        ...DEFAULT_RETRY_SETTINGS,
        policy: POLICY,
        sleep: async () => {},
        random: () => 0,
        log: (line) => lines.push(line),
      },
      "s1 · arm · m3 · #0",
    );

    await r.run(() => generate(r.model, r.signal));

    expect(lines).toEqual([
      "[s1 · arm · m3 · #0] retry: mock-v3/m3 attempt 1/4 failed (429); " +
        "waiting 3000ms (Retry-After), 3000/60000ms of wait budget used",
      "[s1 · arm · m3 · #0] retry: mock-v3/m3 attempt 2/4 failed (500); " +
        "waiting 1000ms, 4000/60000ms of wait budget used",
    ]);
  });

  it("logs nothing without a `log`", async () => {
    const spy = vi.spyOn(console, "log").mockImplementation(() => {});
    try {
      const r = cellRetry(failingV3([apiError(503)]), 60_000, {
        ...DEFAULT_RETRY_SETTINGS,
        policy: POLICY,
        sleep: async () => {},
      });
      await r.run(() => generate(r.model, r.signal));
      expect(r.stats.retries).toBe(1);
      expect(spy).not.toHaveBeenCalled();
    } finally {
      spy.mockRestore();
    }
  });
});

describe("retry settings", () => {
  it("defaults: a8/b2000/c60000/w180000, grace 30000", () => {
    expect(retrySettingsFromEnv({})).toEqual(DEFAULT_RETRY_SETTINGS);
    expect(DEFAULT_RETRY_SETTINGS).toEqual({
      policy: { maxAttempts: 8, baseMs: 2000, maxDelayMs: 60_000, maxTotalWaitMs: 180_000 },
      graceMs: 30_000,
    });
  });

  it("reads every knob from the env", () => {
    expect(
      retrySettingsFromEnv({
        RATEL_LLM_RETRY_MAX_ATTEMPTS: "3",
        RATEL_LLM_RETRY_BASE_MS: "100",
        RATEL_LLM_RETRY_MAX_DELAY_MS: "1000",
        RATEL_LLM_RETRY_MAX_WAIT_MS: "5000",
        RATEL_CELL_TIMEOUT_GRACE_MS: "250",
      }),
    ).toEqual({
      policy: { maxAttempts: 3, baseMs: 100, maxDelayMs: 1000, maxTotalWaitMs: 5000 },
      graceMs: 250,
    });
  });

  it("rejects a non-positive-integer knob with a clear error", () => {
    for (const bad of ["0", "-1", "1.5", "abc", ""]) {
      expect(() => retrySettingsFromEnv({ RATEL_LLM_RETRY_MAX_ATTEMPTS: bad })).toThrow(
        /RATEL_LLM_RETRY_MAX_ATTEMPTS must be a positive integer/,
      );
    }
    expect(() => retrySettingsFromEnv({ RATEL_CELL_TIMEOUT_GRACE_MS: "x" })).toThrow(
      /RATEL_CELL_TIMEOUT_GRACE_MS/,
    );
  });

  it("caps the knobs that become a timer delay at 2^31−1 ms (Node would fire after 1 ms)", () => {
    for (const name of ["RATEL_CELL_TIMEOUT_GRACE_MS", "RATEL_LLM_RETRY_MAX_DELAY_MS"]) {
      expect(() => retrySettingsFromEnv({ [name]: "2147483648" })).toThrow(
        `${name} must be ≤ 2147483647 ms (got "2147483648")`,
      );
      expect(() => retrySettingsFromEnv({ [name]: "2147483647" })).not.toThrow();
    }
    // Not timers: the base is capped by maxDelayMs, the wait budget is a sum.
    expect(
      retrySettingsFromEnv({
        RATEL_LLM_RETRY_BASE_MS: "2147483648",
        RATEL_LLM_RETRY_MAX_WAIT_MS: "2147483648",
      }).policy,
    ).toMatchObject({ baseMs: 2_147_483_648, maxTotalWaitMs: 2_147_483_648 });
  });

  it("labels the recorded policy and echoes the startup line", () => {
    expect(retryPolicyLabel(DEFAULT_RETRY_SETTINGS.policy, 180_000, 30_000)).toBe(
      "a8/b2000/c60000/w180000;timeout=active:180000+g30000",
    );
    expect(retrySettingsLine(DEFAULT_RETRY_SETTINGS, 180_000)).toBe(
      "retry: attempts=8 base=2000ms max-delay=60000ms max-wait=180000ms; " +
        "timeout=180000ms active + 30000ms grace",
    );
  });
});

describe("createBreaker", () => {
  const transient = { error: "Overloaded", error_class: "transient" as const };
  const ok = { error: null };

  it("counts streaks per model; the first abort wins; stopped() prefers fatal", () => {
    const breaker = createBreaker(2);
    breaker.record("A", transient);
    breaker.record("B", transient);
    breaker.record("A", ok); // resets A only
    breaker.record("B", { error: "model b not found" }); // legacy row: access by message
    expect(breaker.isAborted("A")).toBe(false);
    expect(breaker.aborted()).toEqual({
      B: {
        reason: "error_circuit",
        detail: "2 consecutive transient/access errors (last: model b not found)",
      },
    });
    expect(breaker.stopped()).toBe("error_circuit");

    breaker.fatal("B", new Error("later")); // B keeps its first reason
    breaker.fatal("A", new Error("model A not available for this account"));
    expect(breaker.aborted().B.reason).toBe("error_circuit");
    expect(breaker.aborted().A).toEqual({
      reason: "fatal",
      detail: "model A not available for this account",
    });
    expect(breaker.stopped()).toBe("fatal");
  });

  it("request rows neither count toward the streak nor reset it", () => {
    const request = { error: "tools: too many tools", error_class: "request" as const };
    const breaker = createBreaker(2);
    breaker.record("A", request);
    breaker.record("A", request);
    expect(breaker.isAborted("A")).toBe(false);
    expect(breaker.stopped()).toBeUndefined();
    breaker.record("A", transient);
    breaker.record("A", request);
    breaker.record("A", transient);
    expect(breaker.aborted().A?.reason).toBe("error_circuit");
  });

  it("an outage row with no stamped class still counts (legacy message rules)", () => {
    const breaker = createBreaker(1);
    breaker.record("A", { error: "Failed after 3 attempts. Last error: Overloaded" });
    expect(breaker.isAborted("A")).toBe(true);
  });

  it("threshold 0 never trips on a streak; fatal still aborts", () => {
    const breaker = createBreaker(0);
    for (let i = 0; i < 50; i++) breaker.record("A", transient);
    expect(breaker.stopped()).toBeUndefined();
    breaker.fatal("A", "gated");
    expect(breaker.aborted().A).toEqual({ reason: "fatal", detail: "gated" });
  });
});

describe("breaker knob", () => {
  it("RATEL_ABORT_AFTER_CONSECUTIVE_ERRORS: default 10, 0 = off, else a non-negative int", () => {
    expect(breakerThresholdFromEnv({})).toBe(10);
    expect(breakerThresholdFromEnv({ RATEL_ABORT_AFTER_CONSECUTIVE_ERRORS: "0" })).toBe(0);
    expect(breakerThresholdFromEnv({ RATEL_ABORT_AFTER_CONSECUTIVE_ERRORS: "3" })).toBe(3);
    for (const bad of ["-1", "1.5", "x", ""]) {
      expect(() => breakerThresholdFromEnv({ RATEL_ABORT_AFTER_CONSECUTIVE_ERRORS: bad })).toThrow(
        /RATEL_ABORT_AFTER_CONSECUTIVE_ERRORS must be a non-negative integer/,
      );
    }
  });

  it("echoes the knob at startup", () => {
    expect(breakerLine(10)).toBe(
      "breaker: abort a model on a fatal provider error, or after 10 consecutive transient/access errors",
    );
    expect(breakerLine(0)).toMatch(/streak check off \(RATEL_ABORT_AFTER_CONSECUTIVE_ERRORS=0\)/);
  });
});

describe("retry_policy rerun suffix", () => {
  it("appends ';rerun=<label>' when the settings carry the run's rerun policy", async () => {
    const plain = cellRetry(new MockLanguageModelV3(), 1000, DEFAULT_RETRY_SETTINGS);
    const tagged = cellRetry(new MockLanguageModelV3(), 1000, {
      ...DEFAULT_RETRY_SETTINGS,
      rerunLabel: "infra/3",
    });
    try {
      expect(plain.rowFields().retry_policy).toBe(
        "a8/b2000/c60000/w180000;timeout=active:1000+g30000",
      );
      expect(tagged.rowFields().retry_policy).toBe(
        "a8/b2000/c60000/w180000;timeout=active:1000+g30000;rerun=infra/3",
      );
    } finally {
      await plain.run(async () => {});
      await tagged.run(async () => {});
    }
  });
});
