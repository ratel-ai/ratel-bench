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
