import { describe, expect, it } from "vitest";
import type { AtlasTool } from "./atlas-mcp-shim.js";
import {
  type CheckResult,
  checkSandboxCatalog,
  type DoctorProbes,
  doctorExitCode,
  formatResults,
  isPlaceholder,
  runChecks,
} from "./mcpatlas-doctor.js";
import { buildCatalogManifest } from "./mcpatlas-servers.js";

// Includes the three credentialed servers so the credentials check has something
// to find. `scope: "full"` in opts() keeps the size assertion self-consistent;
// the pinned coding-scope count of 79 gets its own test below.
const MANIFEST = buildCatalogManifest("full", {
  github: ["github/get_issue", "github/search_repositories"],
  git: ["git/status"],
  airtable: ["airtable/list_records"],
  "e2b-server": ["e2b-server/run"],
});

function tools(...names: string[]): AtlasTool[] {
  return names.map((name) => ({ name }));
}

const SANDBOX_TOOLS = tools(
  "github_get_issue",
  "github_search_repositories",
  "git_status",
  "airtable_list_records",
  "e2b-server_run",
);

function probes(over: Partial<DoctorProbes> = {}): DoctorProbes {
  return {
    env: { GITHUB_TOKEN: "x", AIRTABLE_API_KEY: "y", E2B_API_KEY: "z" },
    claudeVersion: async () => "2.1.237 (Claude Code)",
    dockerRunning: async () => true,
    composeAvailable: async () => true,
    sandboxHealthy: async () => true,
    sandboxTools: async () => SANDBOX_TOOLS,
    ratelLocal: async () => ({ version: "0.8.1", sdkVersion: "0.9.1" }),
    ...over,
  };
}

function opts(over: Record<string, unknown> = {}) {
  return {
    scope: "full" as const,
    manifest: MANIFEST,
    taskCount: 55,
    taskListHash: "4d25da4975abd735aa",
    expectedTaskListHash: "4d25da4975abd735aa",
    sandboxUrl: "http://localhost:1984",
    ratelLocalPin: "0.8.1",
    probes: probes(),
    ...over,
  };
}

const by = (rs: CheckResult[], name: string) => rs.find((r) => r.name === name);
const blocking = (rs: CheckResult[]) => rs.filter((r) => !r.ok && r.severity === "blocking");

describe("checkSandboxCatalog — the highest-risk assumption in the build", () => {
  it("passes when the sandbox matches the manifest hash", () => {
    const rs = checkSandboxCatalog(SANDBOX_TOOLS, MANIFEST, "http://x");
    expect(blocking(rs)).toEqual([]);
    expect(rs.some((r) => r.name.includes("hash") && r.ok)).toBe(true);
  });

  it("FAILS when the sandbox serves a different tool set than the manifest", () => {
    // The manifest comes from the dataset; the sandbox is a separate source of
    // truth. A drift means the arms would not see the same universe.
    const rs = checkSandboxCatalog(tools("github_get_issue"), MANIFEST, "http://x");
    const f = blocking(rs)[0];
    expect(f).toBeDefined();
    expect(f.detail).toContain("mismatch");
    expect(f.detail).toContain("missing");
  });

  it("names the unexpected tools too", () => {
    const rs = checkSandboxCatalog(
      tools("github_get_issue", "github_search_repositories", "git_status", "git_log"),
      MANIFEST,
      "http://x",
    );
    expect(blocking(rs)[0].detail).toContain("unexpected");
  });

  it("warns, without blocking, when a name does not resolve to a known server", () => {
    // Servers outside the scope are expected in `full`; a naming change is not.
    const rs = checkSandboxCatalog(
      [...SANDBOX_TOOLS, ...tools("weather_now")],
      MANIFEST,
      "http://x",
    );
    const w = rs.find((r) => r.name.includes("naming"));
    expect(w?.severity).toBe("warning");
    expect(w?.fix).toContain("atlas-mcp-shim");
    // the in-scope hash still matches, so nothing blocks
    expect(blocking(rs)).toEqual([]);
  });

  it("fails loudly when the sandbox returns nothing", () => {
    expect(blocking(checkSandboxCatalog(null, MANIFEST, "http://x"))[0].fix).toContain("sandbox");
    expect(blocking(checkSandboxCatalog([], MANIFEST, "http://x"))[0].detail).toContain("zero");
  });
});

describe("runChecks", () => {
  it("passes a healthy environment", async () => {
    const { results } = await runChecks(opts());
    expect(blocking(results)).toEqual([]);
    expect(doctorExitCode(results)).toBe(0);
  });

  it("resolves facts for mcpatlas-run to consume rather than re-probe", async () => {
    const { facts } = await runChecks(opts());
    expect(facts).toMatchObject({
      claude_code_version: "2.1.237 (Claude Code)",
      ratel_local_version: "0.8.1",
      ratel_sdk_version: "0.9.1",
      sandbox_tool_count: 5,
    });
    expect(facts.catalog_sha256).toBe(MANIFEST.catalog_sha256);
  });

  it("names the missing credential rather than failing at call time", async () => {
    const { results } = await runChecks(opts({ probes: probes({ env: { GITHUB_TOKEN: "x" } }) }));
    const c = by(results, "credentials");
    expect(c?.ok).toBe(false);
    expect(c?.detail).toContain("AIRTABLE_API_KEY");
    expect(c?.detail).toContain("E2B_API_KEY");
  });

  it("fails when the agent binary is absent", async () => {
    const { results } = await runChecks(
      opts({ probes: probes({ claudeVersion: async () => null }) }),
    );
    expect(by(results, "claude code")?.ok).toBe(false);
  });

  it("explains the AWS custom-image dockerd trap when the daemon is down", async () => {
    const { results } = await runChecks(
      opts({ probes: probes({ dockerRunning: async () => false }) }),
    );
    const d = by(results, "docker daemon");
    expect(d?.ok).toBe(false);
    expect(d?.fix).toContain("does not auto-start dockerd");
  });

  it("skips docker checks when the caller manages the sandbox", async () => {
    const { results } = await runChecks(
      opts({ requireDocker: false, probes: probes({ dockerRunning: async () => false }) }),
    );
    expect(by(results, "docker daemon")).toBeUndefined();
    expect(doctorExitCode(results)).toBe(0);
  });

  it("does not probe the catalog when the sandbox is unreachable", async () => {
    let called = false;
    const { results, facts } = await runChecks(
      opts({
        probes: probes({
          sandboxHealthy: async () => false,
          sandboxTools: async () => {
            called = true;
            return SANDBOX_TOOLS;
          },
        }),
      }),
    );
    expect(called).toBe(false);
    expect(by(results, "sandbox health")?.ok).toBe(false);
    expect(facts.catalog_sha256).toBeNull();
  });

  it("asserts the PINNED coding server SET, not the tool count", async () => {
    // Server count is stable (it's exactly the 11 coding servers); tool count
    // per server is not — MCP-Atlas's own upstream servers gain and lose tools
    // across versions, so asserting an exact count here would go stale on every
    // such change. The test manifest has 4 servers, not the pinned 11.
    const { results } = await runChecks(opts({ scope: "coding" }));
    const c = by(results, "corpus — catalog size");
    expect(c?.ok).toBe(false);
    expect(c?.detail).toContain("expected 11 servers");
  });

  it("fails on a corpus that is not the pinned one", async () => {
    const { results } = await runChecks(opts({ taskCount: 53 }));
    const c = by(results, "corpus — task count");
    expect(c?.ok).toBe(false);
    expect(c?.detail).toContain("expected 55");
  });

  it("fails on a task-list hash drift", async () => {
    const { results } = await runChecks(opts({ taskListHash: "deadbeef" }));
    expect(by(results, "corpus — task list hash")?.ok).toBe(false);
  });

  it("warns when the transitive SDK version is unresolved", async () => {
    // Ranking lives in the SDK, so an unrecorded version understates what was tested.
    const { results } = await runChecks(
      opts({
        probes: probes({ ratelLocal: async () => ({ version: "0.8.1", sdkVersion: null }) }),
      }),
    );
    const w = by(results, "ratel-local — sdk version");
    expect(w?.severity).toBe("warning");
    expect(doctorExitCode(results)).toBe(0);
  });

  it("fails when the ratel-local pin cannot be resolved", async () => {
    const { results } = await runChecks(opts({ probes: probes({ ratelLocal: async () => null }) }));
    expect(by(results, "ratel-local")?.ok).toBe(false);
    expect(doctorExitCode(results)).toBe(1);
  });
});

describe("isPlaceholder — presence is not enough", () => {
  it("rejects the shapes an unfilled slot actually takes", () => {
    // agent-eval/.env ships GITHUB_TOKEN=... ; AWS SSM ships
    // PLACEHOLDER-set-via-put-parameter. Both would pass a presence check and
    // then fail on the first API call, mid-campaign.
    for (const v of [
      undefined,
      "",
      "   ",
      "...",
      "....",
      "PLACEHOLDER-set-via-put-parameter",
      "xxxx",
      "<your-token>",
    ]) {
      expect(isPlaceholder(v)).toBe(true);
    }
  });

  it("accepts a real-looking value", () => {
    for (const v of ["ghp_abc123", "keyABC.def", "e2b_9f3c"]) {
      expect(isPlaceholder(v)).toBe(false);
    }
  });
});

describe("credentials check", () => {
  it("fails on a placeholder even though the var is present", async () => {
    const { results } = await runChecks(
      opts({
        probes: probes({
          env: { GITHUB_TOKEN: "...", AIRTABLE_API_KEY: "real", E2B_API_KEY: "real" },
        }),
      }),
    );
    const c = by(results, "credentials");
    expect(c?.ok).toBe(false);
    expect(c?.detail).toContain("placeholder value in GITHUB_TOKEN");
  });

  it("distinguishes missing from placeholder in one message", async () => {
    const { results } = await runChecks(opts({ probes: probes({ env: { GITHUB_TOKEN: "..." } }) }));
    const d = by(results, "credentials")?.detail ?? "";
    expect(d).toContain("missing AIRTABLE_API_KEY, E2B_API_KEY");
    expect(d).toContain("placeholder value in GITHUB_TOKEN");
  });
});

describe("probe contract", () => {
  it("accepts a version reported on stderr", async () => {
    // ratel-local --version writes to STDERR, because for an MCP stdio server
    // stdout is the protocol channel. A stdout-only probe reported a valid pin
    // as unresolvable and blocked the run. Verified against 0.8.1 -> sdk 0.9.1.
    const { results, facts } = await runChecks(
      opts({
        probes: probes({ ratelLocal: async () => ({ version: "0.8.1", sdkVersion: "0.9.1" }) }),
      }),
    );
    expect(by(results, "ratel-local")?.ok).toBe(true);
    expect(facts.ratel_local_version).toBe("0.8.1");
    expect(facts.ratel_sdk_version).toBe("0.9.1");
  });
});

describe("exit code and formatting", () => {
  const r = (over: Partial<CheckResult>): CheckResult => ({
    name: "x",
    ok: true,
    severity: "blocking",
    detail: "d",
    ...over,
  });

  it("warnings never fail the run", () => {
    expect(doctorExitCode([r({ ok: false, severity: "warning" })])).toBe(0);
  });

  it("any blocking failure fails the run", () => {
    expect(doctorExitCode([r({}), r({ ok: false })])).toBe(1);
  });

  it("prints the fix under a failure", () => {
    const out = formatResults([
      r({ ok: false, name: "creds", detail: "missing X", fix: "export X" }),
    ]);
    expect(out).toContain("FAIL");
    expect(out).toContain("↳ export X");
    expect(out).toContain("not safe to run");
  });

  it("summarises a clean run", () => {
    expect(formatResults([r({})])).toContain("all checks passed");
  });
});
