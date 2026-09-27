import { describe, expect, it } from "vitest";
import {
  type CellErrorClass,
  cellErrorClass,
  isInfraErrorCell,
  isReusableCell,
} from "./mcpatlas-cell-errors.js";

type Row = { error: string | null; finish_reason: string; judge_error?: string | null };

const row = (r: Row) => ({
  error: r.error,
  finish_reason: r.finish_reason,
  claim_rubric: { judge_error: r.judge_error ?? null },
});

describe("cellErrorClass", () => {
  const cases: Array<[string, Row, CellErrorClass | null]> = [
    ["no error", { error: null, finish_reason: "success" }, null],
    // Claude Code result envelopes (is_error=true, text in `result`)
    [
      "529 overloaded",
      {
        error:
          'API Error: 529 {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}',
        finish_reason: "success",
      },
      "transient",
    ],
    [
      "500 internal",
      { error: "API Error: 500 Internal server error", finish_reason: "success" },
      "transient",
    ],
    [
      "legacy 429 shape",
      { error: "API Error: 429 Too many tokens, please wait", finish_reason: "success" },
      "transient",
    ],
    [
      "connection error",
      { error: "API Error: Connection error.", finish_reason: "success" },
      "transient",
    ],
    [
      "request timed out",
      { error: "API Error: Request timed out.", finish_reason: "success" },
      "transient",
    ],
    ["403", { error: "API Error: 403 Forbidden", finish_reason: "success" }, "access"],
    ["408", { error: "API Error: 408 Request Timeout", finish_reason: "success" }, "transient"],
    ["409", { error: "API Error: 409 Conflict", finish_reason: "success" }, "transient"],
    [
      "content filter (400)",
      {
        error:
          'API Error: 400 {"type":"error","error":{"type":"invalid_request_error","message":"Output blocked by content filtering policy"}}',
        finish_reason: "success",
      },
      "outcome",
    ],
    // Verbatim from the pinned Claude Code 2.1.246 on Bedrock (CLAUDE_CODE_VERSION).
    ...(
      [
        [
          "CC 429 request rejected",
          "API Error: Request rejected (429) · Too many tokens, please wait before trying again.",
          "transient",
        ],
        [
          "CC 500",
          "API Error: 500 Internal server error. This is a server-side issue, usually temporary — try again in a moment. If it persists, check your Amazon Bedrock service status.",
          "transient",
        ],
        [
          "CC 529",
          "API Error: 529 Overloaded. This is a server-side issue, usually temporary — try again in a moment. If it persists, check your Amazon Bedrock service status.",
          "transient",
        ],
        [
          "CC repeated 529",
          "API Error: Repeated 529 Overloaded errors. The API is at capacity — this is usually temporary. Try again in a moment.",
          "transient",
        ],
        [
          "CC 403 auth",
          "Failed to authenticate. API Error: 403 You don't have access to the model with the specified model ID.",
          "access",
        ],
        [
          "CC unable to connect",
          "API Error: Unable to connect to API. Check your internet connection",
          "transient",
        ],
        [
          "CC unable to connect (code)",
          "API Error: Unable to connect to API (ENOTFOUND)",
          "transient",
        ],
        ["CC connection dropped", "API Error: Connection dropped (EPIPE)", "transient"],
        [
          "CC connection refused",
          "API Error: Connection refused — a firewall or proxy may be blocking it (ConnectionRefused)",
          "transient",
        ],
        [
          "CC DNS",
          "API Error: Can't reach the API server — check your internet or DNS (ENOTFOUND)",
          "transient",
        ],
        ["CC no response", "API Error: No response from API", "transient"],
        [
          "CC no response (first byte)",
          "API Error: No response from API within the first-byte window",
          "transient",
        ],
        ["CC bare request timed out", "Request timed out", "transient"],
      ] as const
    ).map(([name, error, cls]): [string, Row, CellErrorClass] => [
      name,
      { error, finish_reason: "success" },
      cls,
    ]),
    // Verbatim from codex-cli 0.153.0 (turn.failed / error events → error_during_execution).
    ...(
      [
        [
          "codex 429 retry limit",
          "exceeded retry limit, last status: 429 Too Many Requests",
          "transient",
        ],
        [
          "codex 502 retry limit",
          "exceeded retry limit, last status: 502 Bad Gateway",
          "transient",
        ],
        [
          "codex 502",
          "unexpected status 502 Bad Gateway: upstream error, url: https://api.openai.com/v1/responses",
          "transient",
        ],
        [
          "codex high demand",
          "We're currently experiencing high demand, which may cause temporary errors.",
          "transient",
        ],
        [
          "codex capacity",
          "Selected model is at capacity. Please try a different model.",
          "transient",
        ],
        [
          "codex stream disconnect",
          "stream disconnected before completion: error sending request for url (https://api.openai.com/v1/responses)",
          "transient",
        ],
        [
          "codex stream decode",
          "stream disconnected before completion: Transport error: network error: error decoding response body",
          "transient",
        ],
        ["codex connection failed", "Connection failed: tcp connect error", "transient"],
        ["codex 401", "unexpected status 401 Unauthorized: invalid api key", "access"],
        ["codex 403", "unexpected status 403 Forbidden: org not verified", "access"],
        ["codex 404", "unexpected status 404 Not Found: model does not exist", "access"],
        ["codex quota", "Quota exceeded. Check your plan and billing details.", "access"],
        [
          "codex usage limit",
          "You've hit your usage limit. Upgrade to Pro (https://chatgpt.com/explore/pro) or try again in 2 days 3 hours.",
          "access",
        ],
        // Retryable errors surfaced once codex's own retries run out.
        [
          "codex rate limit exceeded",
          "rate limit exceeded: Rate limit reached for gpt-5.1 in organization org-AAA on tokens per min (TPM): Limit 30000, Used 22999, Requested 12528. Please try again in 11.054s. Visit https://platform.openai.com/account/rate-limits to learn more.",
          "transient",
        ],
        [
          "codex response read failure",
          "Error while reading the server response: error decoding response body, request id: req_abc",
          "transient",
        ],
        ["codex request timed out", "request timed out", "transient"],
        [
          "codex request timed out (details)",
          "request timed out (while waiting for the first event)",
          "transient",
        ],
        ["codex child timeout", "timeout waiting for child process to exit", "transient"],
        ["codex agent loop died", "internal error; agent loop died unexpectedly", "transient"],
        ["codex io reset", "Connection reset by peer (os error 104)", "transient"],
        ["codex io broken pipe", "Broken pipe (os error 32)", "transient"],
        // A deterministic local Io error is not re-rolled.
        ["codex io ENOENT", "No such file or directory (os error 2)", "outcome"],
        ["codex unknown", "something unexpected happened", "outcome"],
        [
          "codex invalid request",
          '{"error":{"message":"Invalid schema for function","type":"invalid_request_error","code":"invalid_function_parameters"}}',
          "request",
        ],
        [
          "codex context overflow",
          '{"error":{"message":"Your input exceeds the context window","type":"invalid_request_error","code":"context_length_exceeded"}}',
          "outcome",
        ],
      ] as const
    ).map(([name, error, cls]): [string, Row, CellErrorClass] => [
      name,
      { error, finish_reason: "error_during_execution" },
      cls,
    ]),
    [
      "bedrock account gate",
      {
        error:
          "API Error: 400 Model use case details have not been submitted; not available for this account",
        finish_reason: "success",
      },
      "access",
    ],
    [
      "AWS credentials",
      { error: "AWS credential provider failed: no profile", finish_reason: "error" },
      "access",
    ],
    ["other 4xx", { error: "API Error: 422 bad tool schema", finish_reason: "success" }, "request"],
    [
      "prompt too long",
      { error: "API Error: 400 prompt is too long: 210000 tokens", finish_reason: "success" },
      "outcome",
    ],
    // Legacy retry wrapper: classify what it wraps
    [
      "retry wrapper",
      { error: "Failed after 3 attempts. Last error: Overloaded", finish_reason: "error" },
      "transient",
    ],
    // Thrown by runCell (finish_reason "error")
    [
      "sandbox gone",
      {
        error: "catalog integrity: sandbox unreachable at http://localhost:1984",
        finish_reason: "error",
      },
      "transient",
    ],
    [
      "CLI crash on a network drop",
      {
        error:
          "claude produced no parseable result envelope (timedOut=false, exitCode=1, signal=null): fetch failed",
        finish_reason: "error",
      },
      "transient",
    ],
    [
      "CLI killed at the deadline",
      {
        error:
          "claude produced no parseable result envelope (timedOut=true, exitCode=null, signal=SIGKILL): fetch failed",
        finish_reason: "error",
      },
      "timeout",
    ],
    [
      "codex no events on a network drop",
      {
        error:
          "codex produced no parseable events (timedOut=false, exitCode=1, signal=null): fetch failed",
        finish_reason: "error",
      },
      "transient",
    ],
    // A spawn failure (the process never ran), a host-resource errno, or a CLI
    // exit without an envelope: no scorable result, so unrecognised → transient.
    [
      "CLI spawn EAGAIN",
      {
        error:
          "claude produced no parseable result envelope (timedOut=false, exitCode=null, signal=null): \nspawn claude EAGAIN",
        finish_reason: "error",
      },
      "transient",
    ],
    [
      "codex spawn ENOENT",
      {
        error:
          "codex produced no parseable events (timedOut=false, exitCode=null, signal=null): \nspawn codex ENOENT",
        finish_reason: "error",
      },
      "transient",
    ],
    ["sync spawn ENOMEM", { error: "spawn ENOMEM", finish_reason: "error" }, "transient"],
    [
      "ENOSPC",
      { error: "ENOSPC: no space left on device, write", finish_reason: "error" },
      "transient",
    ],
    [
      "CLI exit 1, no envelope",
      {
        error:
          "claude produced no parseable result envelope (timedOut=false, exitCode=1, signal=null): ",
        finish_reason: "error",
      },
      "transient",
    ],
    // Explicit rules still win on a thrown row.
    [
      "CLI crash on context overflow",
      {
        error:
          "claude produced no parseable result envelope (timedOut=false, exitCode=1, signal=null): prompt is too long",
        finish_reason: "error",
      },
      "outcome",
    ],
    // External SIGKILL outside the deadline: the OOM-killer signature.
    [
      "CLI OOM-killed",
      {
        error:
          "claude produced no parseable result envelope (timedOut=false, exitCode=null, signal=SIGKILL): ",
        finish_reason: "error",
      },
      "transient",
    ],
    [
      "CLI killed, with stderr",
      {
        error:
          "claude produced no parseable result envelope (timedOut=false, exitCode=null, signal=SIGKILL): node: warning",
        finish_reason: "error",
      },
      "transient",
    ],
    [
      "codex OOM-killed",
      {
        error:
          "codex produced no parseable events (timedOut=false, exitCode=null, signal=SIGKILL): ",
        finish_reason: "error",
      },
      "transient",
    ],
    [
      "gateway never started",
      {
        error: "ratel cell produced empty telemetry — retrieval data for this cell is lost",
        finish_reason: "error",
      },
      "transient",
    ],
    ["codex deadline", { error: "", finish_reason: "error_timeout" }, "timeout"],
    ["max turns", { error: "", finish_reason: "error_max_turns" }, "outcome"],
    [
      "max turns beats error text",
      { error: "API Error: 529 Overloaded", finish_reason: "error_max_turns" },
      "outcome",
    ],
    // Unrecognised text: a runCell throw left no scorable result (transient);
    // a harness envelope's own failure is the model's outcome.
    ["unknown, thrown", { error: "boom", finish_reason: "error" }, "transient"],
    [
      "unknown, harness envelope",
      { error: "boom", finish_reason: "error_during_execution" },
      "outcome",
    ],
    ["unknown, CC envelope", { error: "boom", finish_reason: "success" }, "outcome"],
  ];

  it.each(cases)("%s", (_name, r, expected) => {
    expect(cellErrorClass(row(r))).toBe(expected);
  });
});

describe("isInfraErrorCell", () => {
  it("is transient|access only", () => {
    expect(
      isInfraErrorCell(row({ error: "API Error: 529 Overloaded", finish_reason: "success" })),
    ).toBe(true);
    expect(
      isInfraErrorCell(row({ error: "API Error: 403 Forbidden", finish_reason: "success" })),
    ).toBe(true);
    expect(isInfraErrorCell(row({ error: "API Error: 422 bad", finish_reason: "success" }))).toBe(
      false,
    );
    expect(isInfraErrorCell(row({ error: "", finish_reason: "error_max_turns" }))).toBe(false);
    expect(isInfraErrorCell(row({ error: null, finish_reason: "success" }))).toBe(false);
  });
});

describe("isReusableCell", () => {
  it("rejects transient|access|request errors and failed judges; keeps final outcomes", () => {
    expect(isReusableCell(row({ error: null, finish_reason: "success" }))).toBe(true);
    expect(isReusableCell(row({ error: "API Error: 422 bad", finish_reason: "success" }))).toBe(
      false,
    );
    expect(isReusableCell(row({ error: "", finish_reason: "error_max_turns" }))).toBe(true);
    expect(
      isReusableCell(row({ error: "API Error: 529 Overloaded", finish_reason: "error_max_turns" })),
    ).toBe(true);
    expect(isReusableCell(row({ error: "", finish_reason: "error_timeout" }))).toBe(true);
    expect(
      isReusableCell(
        row({ error: null, finish_reason: "success", judge_error: "judge failed: Overloaded" }),
      ),
    ).toBe(false);
    expect(
      isReusableCell(
        row({
          error: null,
          finish_reason: "success",
          judge_error: "judge omitted 1 of 2 claim(s)",
        }),
      ),
    ).toBe(true);
    for (const judge_error of [
      "judge truncated at 64 output tokens",
      "judge truncated at the provider's default output limit",
    ]) {
      expect(isReusableCell(row({ error: null, finish_reason: "success", judge_error }))).toBe(
        true,
      );
    }
    expect(
      isReusableCell(
        row({
          error: null,
          finish_reason: "success",
          judge_error: "2 claim(s) need a judge but no model was supplied",
        }),
      ),
    ).toBe(true);
  });

  // runCell's catch writes a thrown message to both `error` and `judge_error`.
  it("keeps a thrown timeout whose judge_error mirrors the error", () => {
    const thrown =
      "claude produced no parseable result envelope (timedOut=true, exitCode=null, signal=SIGKILL): ";
    expect(
      isReusableCell(row({ error: thrown, finish_reason: "error", judge_error: thrown })),
    ).toBe(true);
  });
});
