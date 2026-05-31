import { readFileSync } from "node:fs";
import type { Dataset } from "./types.js";

export function loadDataset(path: string): Dataset {
  const raw = JSON.parse(readFileSync(path, "utf-8")) as Dataset;
  if (!Array.isArray(raw.tools) || !Array.isArray(raw.scenarios)) {
    throw new Error(`dataset at ${path} is missing tools[] or scenarios[]`);
  }
  return raw;
}

export function getExpectedTools(scenarios: Dataset["scenarios"]): string[] {
  const set = new Set<string>();
  for (const s of scenarios) {
    for (const t of s.turns) {
      set.add(t.expectedTool);
    }
  }
  return [...set];
}
