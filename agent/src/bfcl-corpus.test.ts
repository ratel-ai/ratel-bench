// Compatibility smoke test for the BFCL corpora. Proves the normalized JSONL is
// safe for the strict agent layer BEFORE any (expensive) LLM run:
//   1. buildToolBundle over the FULL function universe doesn't throw — i.e. no
//      two distinct tool ids collapse to the same sanitized provider name
//      (the ingest's de-collision guarantee, retrieval/src/ingest/bfcl.rs).
//   2. Every tool's normalized input schema has a string `type` — i.e. BFCL's
//      `dict`/`float`/… vocabulary was rewritten to valid JSON Schema.
//
// The BFCL corpora are gitignored and deleted after a run, so these tests are
// conditional: they run when `test-data/bfcl-*.jsonl` exist (e.g. right after
// `cargo run -p ratel-benchmark-retrieval -- ingest bfcl`), and skip otherwise.

import { existsSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { buildToolBundle, normalizeInputSchema } from "./agents/_shared.js";
import { loadScenarios } from "./corpus.js";
import { resolveRepoPath } from "./paths.js";
import { buildToolUniverse } from "./pool.js";

const CORPORA = [
  { name: "bfcl-simple", path: "test-data/bfcl-simple.jsonl" },
  { name: "bfcl-multiple", path: "test-data/bfcl-multiple.jsonl" },
];

describe("BFCL corpus is registry/provider-safe", () => {
  for (const { name, path } of CORPORA) {
    const abs = resolveRepoPath(path);
    const present = existsSync(abs);
    const maybe = present ? it : it.skip;

    maybe(`${name}: full universe registers without sanitization collisions`, () => {
      const scenarios = loadScenarios(abs);
      const universe = buildToolUniverse(scenarios);
      expect(universe.length).toBeGreaterThan(0);
      // Throws on a post-sanitization name collision; reaching the assertion
      // means the ingest disambiguated every id correctly.
      const bundle = buildToolBundle(universe);
      expect(bundle.activeToolIds.length).toBe(universe.length);
    });

    maybe(`${name}: every tool's input schema normalizes to a typed JSON Schema`, () => {
      const scenarios = loadScenarios(abs);
      const universe = buildToolUniverse(scenarios);
      for (const spec of universe) {
        const schema = normalizeInputSchema(spec.input_schema);
        expect(typeof schema.type).toBe("string");
      }
    });
  }

  if (!CORPORA.some(({ path }) => existsSync(resolveRepoPath(path)))) {
    it.skip("(BFCL corpora absent — run `ingest bfcl` to enable these checks)", () => {});
  }
});
