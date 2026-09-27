import { setTimeout as sleep } from "node:timers/promises";
import {
  APICallError,
  LoadAPIKeyError,
  NoObjectGeneratedError,
  NoSuchModelError,
  RetryError,
} from "ai";
import { describe, expect, it } from "vitest";
import {
  CellTimeoutError,
  classifyError,
  classifyErrorMessage,
  type ErrorClass,
  errorClassOf,
  FatalProviderError,
  isInfraError,
  isRerunnable,
  PROVIDER_ERROR_MARKER,
  RetriesExhaustedError,
  supersede,
} from "./cell-errors.js";

function apiError(statusCode: number, message = `HTTP ${statusCode}`): APICallError {
  return new APICallError({
    message,
    url: "https://api.test/v1",
    requestBodyValues: {},
    statusCode,
  });
}

const OPENAI_QUOTA_MESSAGE =
  "You exceeded your current quota, please check your plan and billing details.";

/** A 429 carrying a parsed OpenAI-style error body (`createJsonErrorResponseHandler` → `data`). */
function quota429(error: Record<string, unknown>): APICallError {
  return new APICallError({
    message: OPENAI_QUOTA_MESSAGE,
    url: "https://api.test/v1",
    requestBodyValues: {},
    statusCode: 429,
    isRetryable: true,
    data: { error: { message: OPENAI_QUOTA_MESSAGE, ...error } },
  });
}

function retryError(lastError: unknown): RetryError {
  return new RetryError({
    message: "Failed after 3 attempts. Last error: …",
    reason: "maxRetriesExceeded",
    errors: [new Error("earlier"), lastError],
  });
}

function marked(kind: "fatal" | "transient"): Error {
  return Object.assign(new Error("AccessDeniedException: not for you"), {
    [PROVIDER_ERROR_MARKER]: { kind, scope: "model", reason: "gated" },
  });
}

/** Shaped like Node's AbortError: the signal's abort reason arrives on `cause`. */
function abortError(reason: unknown): Error {
  return Object.assign(new Error("This operation was aborted", { cause: reason }), {
    name: "AbortError",
  });
}

/** What provider-utils throws when reading a 2xx body fails (drop or abort mid-body). */
function bodyReadFailure(cause: unknown): APICallError {
  return new APICallError({
    message: "Failed to process successful response",
    url: "https://api.test/v1",
    requestBodyValues: {},
    statusCode: 200,
    cause,
  });
}

function cyclic(): Error {
  const err = new Error("boom");
  err.cause = err;
  return err;
}

describe("classifyError maps structured errors", () => {
  const cases: Array<[string, unknown, ErrorClass, { retries?: number }?]> = [
    [
      "RetryError → lastError 400 prompt too long",
      retryError(apiError(400, "prompt is too long: 250000 tokens > 200000 maximum")),
      "outcome",
    ],
    ["RetryError → lastError 503", retryError(apiError(503, "Service Unavailable")), "transient"],
    ["APICallError 403", apiError(403, "Forbidden"), "access"],
    ["APICallError 401", apiError(401, "Unauthorized"), "access"],
    ["APICallError 404", apiError(404, "Not Found"), "access"],
    ["APICallError 424", apiError(424, "Failed Dependency"), "request"],
    ["APICallError 400", apiError(400, "tools: too many tools"), "request"],
    ["APICallError 413", apiError(413, "Payload Too Large"), "request"],
    ["APICallError 429 (retryable)", apiError(429, "Too Many Requests"), "transient"],
    ["APICallError 529 (retryable)", apiError(529, "Overloaded"), "transient"],
    // 2xx body failed the provider schema: SDK marks it non-retryable, but it's a provider hiccup.
    ["APICallError 200 Invalid JSON response", apiError(200, "Invalid JSON response"), "transient"],
    ["APICallError context_length_exceeded", apiError(400, "context_length_exceeded"), "outcome"],
    [
      "Bedrock credential error (plain Error)",
      new Error("AWS credential provider failed: no creds"),
      "access",
    ],
    [
      "plain Error not available for this account",
      new Error("openai.gpt-6-astra is not available for this account."),
      "access",
    ],
    // Bedrock's daily token cap: a retryable 429 no wait inside a cell can clear.
    [
      "APICallError 429 Too many tokens per day (unmarked)",
      apiError(429, "Too many tokens per day, please wait before trying again."),
      "access",
    ],
    // OpenAI's out-of-credit 429: retryable per the SDK, but no backoff can outwait it.
    [
      "APICallError 429 insufficient_quota (OpenAI)",
      quota429({ type: "insufficient_quota", code: "insufficient_quota" }),
      "access",
    ],
    [
      "APICallError 429 insufficient_quota in the raw body only",
      new APICallError({
        message: OPENAI_QUOTA_MESSAGE,
        url: "https://api.test/v1",
        requestBodyValues: {},
        statusCode: 429,
        isRetryable: true,
        responseBody: '{"error":{"code":"insufficient_quota"}}',
      }),
      "access",
    ],
    // Gemini-compatible endpoints reuse the sentence for per-minute limits: the code decides.
    [
      "APICallError 429 quota message without insufficient_quota stays transient",
      quota429({ code: 429, status: "RESOURCE_EXHAUSTED" }),
      "transient",
    ],
    [
      "OpenAI rate limit 429 stays transient",
      apiError(
        429,
        "Rate limit reached for gpt-5.4-mini in organization org-x on tokens per min (TPM): Limit 200000, Used 199000",
      ),
      "transient",
    ],
    // Without the content-filter rule running before the status rules this would be `request`.
    [
      "content filter (Anthropic 400)",
      apiError(400, "Output blocked by content filtering policy"),
      "outcome",
    ],
    ["fatal marker", marked("fatal"), "access"],
    // The retry wrapper's own errors (llm-retry.ts): both are non-APICallError.
    [
      "RetriesExhaustedError → its cause (503)",
      new RetriesExhaustedError(apiError(503, "Service Unavailable"), 8),
      "transient",
    ],
    [
      "RetriesExhaustedError fail-fast on Retry-After → its cause (429)",
      new RetriesExhaustedError(apiError(429), 1, "Retry-After 90000ms > max delay 60000ms"),
      "transient",
    ],
    ["FatalProviderError (wrapping a 403)", new FatalProviderError(apiError(403)), "access"],
    [
      "FatalProviderError (wrapping a fatal-marked 429)",
      new FatalProviderError(
        Object.assign(apiError(429, "Too many tokens per day"), {
          [PROVIDER_ERROR_MARKER]: { kind: "fatal", scope: "model", reason: "daily cap" },
        }),
      ),
      "access",
    ],
    ["transient marker", marked("transient"), "transient"],
    // Marker precedence: the messages below match no message rule, so only the
    // marker (checked before message and status rules) decides.
    [
      "fatal marker wins over a retryable APICallError",
      Object.assign(apiError(429, "Too Many Requests"), {
        [PROVIDER_ERROR_MARKER]: { kind: "fatal", scope: "model", reason: "daily cap" },
      }),
      "access",
    ],
    [
      "transient marker wins over a 403 (Bedrock ExpiredTokenException)",
      Object.assign(apiError(403, "ExpiredTokenException: token expired"), {
        [PROVIDER_ERROR_MARKER]: { kind: "transient", scope: "model", reason: "expired token" },
      }),
      "transient",
    ],
    [
      "transient marker wins over an access message rule",
      Object.assign(new Error("model foo not found"), {
        [PROVIDER_ERROR_MARKER]: { kind: "transient", scope: "model", reason: "flaky lookup" },
      }),
      "transient",
    ],
    ["CellTimeoutError", new CellTimeoutError(180000), "timeout"],
    ["CellTimeoutError with retries", new CellTimeoutError(180000), "transient", { retries: 1 }],
    [
      "AbortError whose cause is a CellTimeoutError",
      abortError(new CellTimeoutError(1000)),
      "timeout",
    ],
    ["AbortError with an unrelated cause", abortError(new Error("user cancelled")), "outcome"],
    [
      "NoObjectGeneratedError",
      new NoObjectGeneratedError({
        message: "No object generated: could not parse the response.",
        response: { id: "r", timestamp: new Date(0), modelId: "m" },
        usage: {
          inputTokens: 1,
          outputTokens: 1,
          totalTokens: 2,
          inputTokenDetails: { noCacheTokens: 1, cacheReadTokens: 0, cacheWriteTokens: 0 },
          outputTokenDetails: { textTokens: 1, reasoningTokens: 0 },
        },
        finishReason: "length",
      }),
      "outcome",
    ],
    ["LoadAPIKeyError", new LoadAPIKeyError({ message: "Anthropic API key is missing" }), "access"],
    [
      "NoSuchModelError",
      new NoSuchModelError({ modelId: "nope", modelType: "languageModel" }),
      "access",
    ],
    ["network: fetch failed", new TypeError("fetch failed"), "transient"],
    ["network: ECONNRESET", new Error("read ECONNRESET"), "transient"],
    ["network: undici", new Error("UND_ERR_SOCKET"), "transient"],
    ["network: other side closed", new Error("other side closed"), "transient"],
    [
      "mid-body socket drop wrapped by the SDK",
      bodyReadFailure(
        new TypeError("terminated", {
          cause: Object.assign(new Error("other side closed"), {
            name: "SocketError",
            code: "UND_ERR_SOCKET",
          }),
        }),
      ),
      "transient",
    ],
    [
      "mid-body ECONNRESET wrapped by the SDK",
      bodyReadFailure(new Error("read ECONNRESET")),
      "transient",
    ],
    [
      "mid-body undici code with a non-network message",
      bodyReadFailure(
        Object.assign(new Error("Body Timeout Error"), { code: "UND_ERR_BODY_TIMEOUT" }),
      ),
      "transient",
    ],
    [
      "mid-body abort by a CellTimeoutError",
      bodyReadFailure(new CellTimeoutError(1000)),
      "timeout",
    ],
    [
      "mid-body abort by a CellTimeoutError with retries",
      bodyReadFailure(new CellTimeoutError(1000)),
      "transient",
      { retries: 1 },
    ],
    ["mid-body failure with an unrelated cause", bodyReadFailure(new Error("boom")), "outcome"],
    [
      "4xx with a network-looking cause stays request",
      Object.assign(apiError(400, "bad request"), { cause: new Error("other side closed") }),
      "request",
    ],
    ["cyclic cause chain", cyclic(), "outcome"],
    ["network: thrown string", "fetch failed", "transient"],
    ["unknown Error", new Error("boom"), "outcome"],
    ["non-Error value", "boom", "outcome"],
  ];

  it.each(cases)("%s", (_name, err, expected, opts) => {
    expect(classifyError(err, opts)).toBe(expected);
  });

  it("classifies a real Node abort by a CellTimeoutError as timeout", async () => {
    const abortedBy = async (reason: unknown) => {
      const ac = new AbortController();
      const pending = sleep(10_000, null, { signal: ac.signal });
      ac.abort(reason);
      return pending.catch((e: unknown) => e);
    };
    expect(classifyError(await abortedBy(new CellTimeoutError(1000)))).toBe("timeout");
    expect(classifyError(await abortedBy(new Error("user cancelled")))).toBe("outcome");
  });

  it("agrees with the legacy classifier on an Invalid JSON response", () => {
    const live = apiError(200, "Invalid JSON response");
    expect(classifyError(live)).toBe(classifyErrorMessage(live.message));
  });
});

describe("classifyErrorMessage classifies every audited S3 string", () => {
  // Every distinct `error` string in the S3 copy of raw/bfcl/agent*.jsonl and
  // raw/sragents/agent*.jsonl (audit 2026-09-26), pinned to its class.
  const audited: Array<[string, ErrorClass]> = [
    ["No object generated: could not parse the response.", "outcome"],
    ["No object generated: response did not match schema.", "outcome"],
    ["Failed after 3 attempts. Last error: Internal server error", "transient"],
    ["Failed after 3 attempts. Last error: Overloaded", "transient"],
    ["Failed after 3 attempts. Last error: Service Unavailable", "transient"],
    ["Failed after 3 attempts. Last error: Cannot connect to API: other side closed", "transient"],
    ["run timed out after 180000ms", "timeout"],
    ["run timed out after 120000ms", "timeout"],
    ["run timed out after 60000ms", "timeout"],
    ["Output blocked by content filtering policy", "outcome"],
    ["model 'qwen2.5:3b' not found", "access"],
    ["Invalid JSON response", "transient"],
    [
      "openai.gpt-6-astra is not available for this account. You can explore other available models on Amazon Bedrock. For additional access options, contact AWS Sales at https://aws.amazon.com/contact-us/sales-support/",
      "access",
    ],
  ];

  it("a legacy daily-cap row is access", () => {
    expect(
      classifyErrorMessage(
        "Failed after 3 attempts. Last error: Too many tokens per day, please wait before trying again.",
      ),
    ).toBe("access");
  });

  it("a legacy OpenAI out-of-credit row is access", () => {
    expect(
      classifyErrorMessage(`Failed after 3 attempts. Last error: ${OPENAI_QUOTA_MESSAGE}`),
    ).toBe("access");
  });

  it.each(audited)("%s", (message, expected) => {
    expect(classifyErrorMessage(message)).toBe(expected);
  });

  it("classifies the last error a 'Failed after N attempts' message wraps", () => {
    expect(classifyErrorMessage("Failed after 3 attempts. Last error: prompt is too long")).toBe(
      "outcome",
    );
    expect(classifyErrorMessage("Failed after 3 attempts. Last error: boom")).toBe("outcome");
    // RetriesExhaustedError's fail-fast form carries its reason in parentheses.
    expect(
      classifyErrorMessage(
        "Failed after 1 attempts (Retry-After 90000ms > max delay 60000ms). Last error: Overloaded",
      ),
    ).toBe("transient");
    // Both fail-fast reasons: only recursion past the parenthesised reason gets these.
    for (const reason of [
      "Retry-After 90000ms > max delay 60000ms",
      "retry wait budget 60000ms: next wait 2000ms > 500ms left",
    ]) {
      expect(
        classifyErrorMessage(
          `Failed after 1 attempts (${reason}). Last error: run timed out after 5ms`,
        ),
      ).toBe("timeout");
    }
    // The timeout rule is the only anchored one (startsWith), so it is the case
    // where only recursion into the wrapped message gets the class right.
    expect(
      classifyErrorMessage("Failed after 3 attempts. Last error: run timed out after 1000ms"),
    ).toBe("timeout");
  });

  it("recurses into a non-retryable RetryError message", () => {
    expect(
      classifyErrorMessage("Failed after 2 attempts with non-retryable error: 'Overloaded'"),
    ).toBe("transient");
    expect(
      classifyErrorMessage(
        "Failed after 2 attempts with non-retryable error: 'Input is too long for requested model.'",
      ),
    ).toBe("outcome");
    expect(
      classifyErrorMessage(
        "Failed after 2 attempts with non-retryable error: 'run timed out after 1000ms'",
      ),
    ).toBe("timeout");
  });

  it("falls back to outcome for unknown strings", () => {
    expect(classifyErrorMessage("something odd")).toBe("outcome");
  });
});

describe("row predicates", () => {
  const ok = { error: null };
  const legacyTransient = { error: "Failed after 3 attempts. Last error: Overloaded" };
  const stamped = (error_class: ErrorClass) => ({ error: "x", error_class });

  it("errorClassOf prefers the stamped class, falls back to the message, null when not errored", () => {
    expect(errorClassOf(ok)).toBeNull();
    expect(errorClassOf(legacyTransient)).toBe("transient");
    expect(errorClassOf({ error: "Overloaded", error_class: "request" })).toBe("request");
  });

  it("isInfraError is transient|access only", () => {
    expect(isInfraError(ok)).toBe(false);
    expect(isInfraError(stamped("transient"))).toBe(true);
    expect(isInfraError(stamped("access"))).toBe(true);
    expect(isInfraError(stamped("request"))).toBe(false);
    expect(isInfraError(stamped("timeout"))).toBe(false);
    expect(isInfraError(stamped("outcome"))).toBe(false);
    expect(isInfraError(legacyTransient)).toBe(true);
  });

  it("isRerunnable policy matrix", () => {
    const classes: ErrorClass[] = ["transient", "access", "request", "timeout", "outcome"];
    const expected: Record<"infra" | "all" | "none", boolean[]> = {
      infra: [true, true, true, false, false],
      all: [true, true, true, true, true],
      none: [false, false, false, false, false],
    };
    for (const policy of ["infra", "all", "none"] as const) {
      expect(classes.map((c) => isRerunnable(stamped(c), policy))).toEqual(expected[policy]);
      expect(isRerunnable(ok, policy)).toBe(false);
    }
    // `infra` is the default policy.
    expect(isRerunnable(stamped("request"))).toBe(true);
    expect(isRerunnable(stamped("timeout"))).toBe(false);
  });

  it("supersede keeps last non-rerunnable row per key, else last row", () => {
    const rows = [
      { k: "a", error: "Overloaded", n: 1 },
      { k: "b", error: null, n: 2 },
      { k: "a", error: null, n: 3 },
      { k: "b", error: "run timed out after 1000ms", n: 4 },
      { k: "a", error: "Failed after 3 attempts. Last error: Internal server error", n: 5 },
      { k: "c", error: "Overloaded", n: 6 },
      { k: "c", error_class: "access" as ErrorClass, error: "x", n: 7 },
    ];
    // a: last non-rerunnable is n3 (n5 is transient); b: n4 (timeout is final);
    // c: only rerunnable rows → the last one, n7. Output follows file order.
    expect(supersede(rows, (r) => r.k).map((r) => r.n)).toEqual([3, 4, 7]);
  });

  it("supersede keeps file order, not the order keys were first seen", () => {
    const interleaved = [
      { k: "a", error: "Overloaded", n: 1 },
      { k: "b", error: null, n: 2 },
      { k: "a", error: null, n: 3 },
    ];
    // a is seen first but its kept row (n3) comes after b's (n2).
    expect(supersede(interleaved, (r) => r.k).map((r) => r.n)).toEqual([2, 3]);
  });
});
