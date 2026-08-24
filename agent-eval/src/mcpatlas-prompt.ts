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
 * Appended to Claude Code's SYSTEM prompt (`--append-system-prompt`), not the
 * task prompt above — a different layer, same frozen/hashed/both-arms-identical
 * treatment as `--strict-mcp-config` or `--disallowed-tools`.
 *
 * This is a statement of fact about the environment, not task strategy: a live
 * run showed the agent try Bash/Grep/Read/Glob (correctly disallowed), get
 * rejected, and conclude "no tools are available in this session" — never
 * trying its actual registered tools at all. This corrects that false belief
 * without saying anything about HOW to use the real tools, still less "search
 * first" — the exact scaffolding `buildPrompt`'s docstring warns against,
 * since that would coach the ratel arm on the mechanism under test. "Bash is
 * unavailable" is true and identical on both arms; "search before you invoke"
 * would not be.
 */
export const SYSTEM_PROMPT_ADDENDUM =
  "Note: shell and direct file-access tools (Bash, Read, Grep, Glob, and similar) " +
  "are not available in this session. You do have other tools registered — use " +
  "those to complete the task.";

export const SYSTEM_PROMPT_ADDENDUM_HASH = createHash("sha256")
  .update(SYSTEM_PROMPT_ADDENDUM)
  .digest("hex");
