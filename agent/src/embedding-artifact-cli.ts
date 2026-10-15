import { parseArgs } from "node:util";
import { buildEmbeddingArtifact, UnsupportedEmbeddingArtifactError } from "./embedding-artifact.js";
import { selectVersion } from "./sdk/resolve.js";

async function main(): Promise<void> {
  const { values, positionals } = parseArgs({
    allowPositionals: true,
    options: {
      benchmark: { type: "string" },
      "sdk-version": { type: "string", default: "" },
      output: { type: "string" },
      corpus: { type: "string" },
      skills: { type: "string" },
    },
  });
  if (positionals.length !== 1 || positionals[0] !== "build") {
    throw new Error(
      "Usage: embedding-artifact build --benchmark bfcl|sragents --sdk-version X --output <file>",
    );
  }
  if (values.benchmark !== "bfcl" && values.benchmark !== "sragents") {
    throw new Error("--benchmark must be bfcl or sragents");
  }
  if (!values.output) throw new Error("--output is required");
  selectVersion(values["sdk-version"]);
  await buildEmbeddingArtifact({
    benchmark: values.benchmark,
    output: values.output,
    corpus: values.corpus,
    skills: values.skills,
  });
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = error instanceof UnsupportedEmbeddingArtifactError ? 3 : 1;
});
