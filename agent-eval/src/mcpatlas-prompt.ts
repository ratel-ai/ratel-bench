// The frozen task prompt. Its own file so PROMPT_HASH moves only when the text
// moves, and so that change is a one-line, reviewable diff.

import { createHash } from "node:crypto";

export const PROMPT_ID = "mcpatlas-v1-verbatim";

/**
 * MCP-Atlas prompts are self-contained and deliberately name no tool or server —
 * that omission is the benchmark's retrieval challenge. Adding scaffolding
 * ("consider searching for a tool first...") would coach the ratel arm and
 * destroy the comparison, so the task prompt is passed through untouched.
 */
export function buildPrompt(taskPrompt: string): string {
  return taskPrompt;
}

export const PROMPT_HASH = createHash("sha256")
  .update(`${PROMPT_ID}\n${buildPrompt.toString()}`)
  .digest("hex");

/**
 * Appended to Claude Code's own system prompt (`--append-system-prompt`), on
 * every cell of both arms. Not a flag: it is part of the frozen configuration,
 * hashed into `system_prompt_addendum_hash` like `prompt_hash`.
 *
 * WHY. We do not set a system prompt, which means Claude Code supplies its own
 * — a full interactive-coding-assistant persona. Upstream MCP-Atlas runs an
 * EMPTY system prompt over a bare completion loop. That difference shows up as
 * the single largest failure bucket in this mode: 13 of 30 cells at k=3 made
 * ZERO tool calls, ending after 1-2 turns with a clarifying question
 * ("I can help you... 1. Where is the data? 2. What format?"). Both harnesses
 * end the episode when the model returns no tool calls, so asking is fatal —
 * but only ours has a persona trained to ask when a request is underspecified.
 *
 * Every ambiguity that triggered it was resolvable by looking: the slackr repo,
 * `Barber Shop.csv`, and the pet-care/food-and-beverage CSVs are all sitting in
 * /data, one `list_directory` away.
 *
 * WHAT IT DELIBERATELY DOES NOT SAY. Nothing about which tools exist, nothing
 * about searching before invoking, nothing about the gateway. That would coach
 * the ratel arm on the exact mechanism under test — the hazard
 * `buildPrompt`'s docstring above warns about. This states one fact about the
 * episode format, equally true on both arms.
 *
 * An earlier attempt (reverted, b8641ac) instead told the model which built-ins
 * were unavailable. It backfired: the model quoted the restriction back as its
 * reason for giving up ("this session doesn't have Bash access — so I can't
 * search git history"). Naming what is missing invites surrender; naming that
 * nobody will answer does not.
 */
export const SYSTEM_PROMPT_ADDENDUM =
  "You are running autonomously. No user is available to answer questions or to " +
  "provide files — if you ask for clarification, the session ends with the task " +
  "unfinished. When information is missing, find it yourself using the tools " +
  "available to you.";

export const SYSTEM_PROMPT_ADDENDUM_HASH = createHash("sha256")
  .update(SYSTEM_PROMPT_ADDENDUM)
  .digest("hex");
