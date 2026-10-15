import { parseArgs } from "node:util";
import {
  buildEmbeddingArtifact,
  embeddingArtifactCapabilities,
  UnsupportedEmbeddingArtifactError,
} from "./embedding-artifact.js";
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
  const command = positionals[0];
  if (positionals.length !== 1 || (command !== "build" && command !== "capabilities")) {
    throw new Error(
      "Usage: embedding-artifact build --benchmark bfcl|sragents --sdk-version X --output <file>\n" +
        "       embedding-artifact capabilities --sdk-version X",
    );
  }
  selectVersion(values["sdk-version"]);
  if (command === "capabilities") {
    console.log(JSON.stringify(await embeddingArtifactCapabilities()));
    return;
  }
  if (values.benchmark !== "bfcl" && values.benchmark !== "sragents") {
    throw new Error("--benchmark must be bfcl or sragents");
  }
  if (!values.output) throw new Error("--output is required");
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
