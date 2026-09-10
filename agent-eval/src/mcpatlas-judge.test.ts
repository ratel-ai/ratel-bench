import { describe, expect, it, vi } from "vitest";
import { screenClaim } from "./mcpatlas-claim-match.js";
import {
  buildJudgePrompt,
  calibration,
  judgeClaims,
  SYSTEM,
  screenScore,
  verdictFor,
} from "./mcpatlas-judge.js";

const MODEL = { modelId: "claude-sonnet-4-6" } as never;

/** Stand-in for generateObject that returns fixed scores for whatever it is asked. */
function fakeJudge(scoreFor: (index: number) => 0 | 0.5 | 1, spy?: { prompt?: string }) {
  return (async (opts: { prompt: string }) => {
    if (spy) spy.prompt = opts.prompt;
    const indices = [...opts.prompt.matchAll(/^\[(\d+)\]/gm)].map((m) => Number(m[1]));
    return {
      object: {
        scores: indices.map((i) => ({
          claim_index: i,
          score: scoreFor(i),
          evidence: "quoted span",
          explanation: "because",
        })),
      },
      usage: { inputTokens: 100, outputTokens: 20 },
    };
  }) as never;
}

describe("verdictFor — computed in code, never asserted by the model", () => {
  it("uses MCP-Atlas's 75% bar", () => {
    expect(verdictFor(0.75)).toBe("pass");
    expect(verdictFor(0.74)).toBe("partial");
    expect(verdictFor(0.4)).toBe("partial");
    expect(verdictFor(0.39)).toBe("fail");
  });

  it("null coverage is n/a, never a fail", () => {
    expect(verdictFor(null)).toBe("n/a");
  });
});

describe("judge prompt", () => {
  it("never leaks the trajectory, gold tools, or the arm label", () => {
    const p = buildJudgePrompt("do it", [{ index: 0, text: "claim one" }], "the answer");
    for (const leak of ["native", "ratel", "gold", "trajectory", "tool_call"]) {
      expect(p.toLowerCase()).not.toContain(leak);
    }
    expect(SYSTEM).toContain("You do NOT see which tools were called");
  });

  it("labels claims with the index the model must echo back", () => {
    expect(buildJudgePrompt("t", [{ index: 3, text: "c" }], "a")).toContain("[3] c");
  });

  it("says so explicitly when the agent produced no text", () => {
    expect(buildJudgePrompt("t", [], "")).toContain("no final text");
  });
});

describe("judgeClaims — screen first, model for the residual", () => {
  const task = "find the creation year";

  it("costs nothing when the screen decides everything", async () => {
    const generate = vi.fn(fakeJudge(() => 1));
    const r = await judgeClaims({
      taskId: "t1",
      prompt: task,
      claims: ["The repo was created in 2013."],
      finalText: "It was created in 2013.",
      model: MODEL,
      generate,
    });
    expect(generate).not.toHaveBeenCalled();
    expect(r.claims_sent_to_llm).toBe(0);
    expect(r.auto_rate).toBe(1);
    expect(r.coverage).toBe(1);
    expect(r.verdict).toBe("pass");
    expect(r.scored_by).toEqual(["screen"]);
  });

  it("sends ONLY the ambiguous claims to the model", async () => {
    const spy: { prompt?: string } = {};
    const r = await judgeClaims({
      taskId: "t2",
      prompt: task,
      claims: [
        "The repo was created in 2013.", // screen: supported
        "The maintainer is responsive.", // screen: no atom -> ambiguous
      ],
      finalText: "It was created in 2013.",
      model: MODEL,
      generate: fakeJudge(() => 1, spy),
    });
    expect(r.claims_auto_scored).toBe(1);
    expect(r.claims_sent_to_llm).toBe(1);
    expect(spy.prompt).toContain("[1] The maintainer is responsive.");
    expect(spy.prompt).not.toContain("[0]");
    expect(r.scored_by).toEqual(["screen", "llm"]);
    expect(r.coverage).toBe(1);
  });

  it("mixes screen and model scores into one coverage figure", async () => {
    const r = await judgeClaims({
      taskId: "t3",
      prompt: task,
      claims: [
        "The repo was created in 2013.", // screen -> 1
        "The tone is friendly.", // llm -> 0.5
      ],
      finalText: "It was created in 2013.",
      model: MODEL,
      generate: fakeJudge(() => 0.5),
    });
    expect(r.claims.map((c) => c.score)).toEqual([1, 0.5]);
    expect(r.coverage).toBe(0.75);
    expect(r.verdict).toBe("pass");
  });

  it("judgeAll overrides the screen, for calibration", async () => {
    const spy: { prompt?: string } = {};
    const r = await judgeClaims({
      taskId: "t4",
      prompt: task,
      claims: ["The repo was created in 2013."],
      finalText: "It was created in 2013.",
      model: MODEL,
      judgeAll: true,
      generate: fakeJudge(() => 0, spy),
    });
    expect(spy.prompt).toContain("[0]");
    expect(r.scored_by).toEqual(["llm"]);
    expect(r.claims[0].score).toBe(0);
  });

  it("a judge failure is n/a with null coverage — NEVER a silent fail", async () => {
    const r = await judgeClaims({
      taskId: "t5",
      prompt: task,
      claims: ["The tone is friendly."],
      finalText: "hello",
      model: MODEL,
      generate: (async () => {
        throw new Error("provider exploded");
      }) as never,
    });
    expect(r.verdict).toBe("n/a");
    expect(r.coverage).toBeNull();
    expect(r.judge_error).toContain("provider exploded");
  });

  it("reports when the model skips claims it was asked to score", async () => {
    const r = await judgeClaims({
      taskId: "t6",
      prompt: task,
      claims: ["The tone is friendly.", "The style is terse."],
      finalText: "hello",
      model: MODEL,
      generate: (async () => ({
        object: { scores: [{ claim_index: 0, score: 1, evidence: "", explanation: "" }] },
        usage: { inputTokens: 1, outputTokens: 1 },
      })) as never,
    });
    expect(r.judge_error).toContain("omitted 1");
    expect(r.coverage).toBeNull();
  });

  it("runs screen-only with no model, leaving the residual unscored", async () => {
    const r = await judgeClaims({
      taskId: "t7",
      prompt: task,
      claims: ["The repo was created in 2013.", "The tone is friendly."],
      finalText: "It was created in 2013.",
    });
    expect(r.claims_auto_scored).toBe(1);
    expect(r.scored_by).toEqual(["screen", "unscored"]);
    // Deliberately null, not 0: "could not score" and "was wrong" differ.
    expect(r.coverage).toBeNull();
    expect(r.verdict).toBe("n/a");
    expect(r.judge_error).toContain("no model was supplied");
  });

  it("meters judge tokens separately so they cannot pollute arm cost", async () => {
    const r = await judgeClaims({
      taskId: "t8",
      prompt: task,
      claims: ["The tone is friendly."],
      finalText: "hello",
      model: MODEL,
      generate: fakeJudge(() => 1),
    });
    expect(r.judge_input_tokens).toBe(100);
    expect(r.judge_output_tokens).toBe(20);
  });
});

describe("calibration", () => {
  it("measures screen-vs-model agreement on claims the screen would have decided", async () => {
    const r = await judgeClaims({
      taskId: "c1",
      prompt: "p",
      claims: [
        "The repo was created in 2013.", // screen: supported (1)
        "The commit is 266f49303621cf01979ed65a02e1f92c1da09460.", // screen: unsupported (0)
      ],
      finalText: "It was created in 2013.",
      model: MODEL,
      judgeAll: true,
      generate: fakeJudge((i) => (i === 0 ? 1 : 1)), // model disagrees on claim 1
    });
    const c = calibration(r);
    expect(c.rows).toHaveLength(2);
    expect(c.rows[0].agrees).toBe(true);
    expect(c.rows[1].agrees).toBe(false);
    expect(c.agreement).toBe(0.5);
  });

  it("has nothing to report when the screen was not overridden", async () => {
    const r = await judgeClaims({
      taskId: "c2",
      prompt: "p",
      claims: ["The repo was created in 2013."],
      finalText: "created in 2013",
      model: MODEL,
      generate: fakeJudge(() => 1),
    });
    expect(calibration(r).agreement).toBeNull();
  });
});

describe("screenScore", () => {
  it("maps the two confident screen verdicts to 1 and 0", () => {
    expect(screenScore(screenClaim("created in 2013", "x#0", "created in 2013"))).toBe(1);
    expect(screenScore(screenClaim("commit 266f49303621cf0197", "x#0", "nothing"))).toBe(0);
  });
});
