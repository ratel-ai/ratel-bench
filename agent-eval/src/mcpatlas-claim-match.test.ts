import { describe, expect, it } from "vitest";
import {
  atomPresent,
  extractAtoms,
  normalizeNumber,
  normalizeText,
  screenClaim,
  screenClaims,
} from "./mcpatlas-claim-match.js";

const v = (claim: string, answer: string) => screenClaim(claim, "t#0", answer).verdict;

describe("normalization", () => {
  it("expands small number words so 'seven' and '7' compare equal", () => {
    expect(normalizeText("the difference is seven")).toContain(" 7 ");
  });

  it("strips currency and thousands separators", () => {
    expect(normalizeNumber("$1,296.16")).toBe("1296.16");
    expect(normalizeNumber("296.160")).toBe("296.16");
    expect(normalizeNumber("-7.4")).toBe("-7.4");
  });
});

describe("extractAtoms", () => {
  it("keeps a commit hash whole rather than shredding it into digits", () => {
    const atoms = extractAtoms("the commit is 266f49303621cf01979ed65a02e1f92c1da09460");
    expect(atoms.filter((a) => a.kind === "hash")).toHaveLength(1);
    expect(atoms.some((a) => a.kind === "number")).toBe(false);
  });

  it("pulls backticked code spans", () => {
    const atoms = extractAtoms("it uses `@radix-ui/react-aspect-ratio`");
    expect(atoms[0]).toMatchObject({ kind: "code", strength: "strong" });
  });

  it("pulls quoted titles", () => {
    const atoms = extractAtoms('the issue is titled "New fart is triggered"');
    expect(atoms.some((a) => a.kind === "quoted" && a.text.includes("New fart"))).toBe(true);
  });

  it("pulls owner/repo paths and scoped packages", () => {
    expect(extractAtoms("in theonion/fartscroll.js").some((a) => a.strength === "strong")).toBe(
      true,
    );
  });

  it("pulls CamelCase names", () => {
    expect(extractAtoms("the AspectRatio component").some((a) => a.text === "AspectRatio")).toBe(
      true,
    );
  });

  it("does NOT treat a sentence-initial capital as an identifier", () => {
    // Every claim is a sentence, so "The"/"There" as strong atoms would give
    // almost every claim a bogus atom that any prose answer satisfies.
    for (const word of ["The", "There", "This", "Issue"]) {
      const atoms = extractAtoms(`${word} thing happened.`);
      expect(atoms.map((a) => a.text)).not.toContain(word);
    }
  });

  it("grades a hyphenated slug as medium, since English compounds share the shape", () => {
    const atoms = extractAtoms("balldontlie-mcp is sports-related");
    expect(atoms.find((a) => a.text === "balldontlie-mcp")?.strength).toBe("medium");
    // medium can support a claim but must never alone declare one unsupported
    expect(atoms.every((a) => a.strength !== "strong")).toBe(true);
  });

  it("grades number strength — a small integer is weak, a decimal is not", () => {
    expect(extractAtoms("5 movies")[0].strength).toBe("weak");
    expect(extractAtoms("$296.16 in sales")[0].strength).toBe("medium");
    expect(extractAtoms("created in 2013")[0].strength).toBe("medium");
  });
});

describe("screenClaim — auto-decides only where a wrong call is implausible", () => {
  it("supports a claim whose strong atoms all appear", () => {
    expect(
      v("The repository was created in 2013.", "The repo was created in 2013, per GitHub."),
    ).toBe("supported");
  });

  it("marks unsupported when a strong atom is entirely absent", () => {
    expect(
      v("The commit is 266f49303621cf01979ed65a02e1f92c1da09460.", "I could not determine that."),
    ).toBe("unsupported");
  });

  it("matches an abbreviated commit hash", () => {
    expect(
      v("The commit is 266f49303621cf01979ed65a02e1f92c1da09460.", "The commit was 266f4930."),
    ).toBe("supported");
  });

  it("refuses to decide when a negation cue sits near the atom", () => {
    // "not created in 2013" contains 2013 — presence means the opposite here.
    expect(v("The repository was created in 2013.", "The repo was not created in 2013.")).toBe(
      "ambiguous",
    );
  });

  it("refuses to decide on a lone weak number — it could be coincidence", () => {
    expect(v("There are 5 movies.", "I looked at 5 different things.")).toBe("ambiguous");
  });

  it("refuses to decide when a weak atom is merely absent — absence proves nothing", () => {
    expect(v("There are 5 movies.", "The answer is unrelated prose.")).toBe("ambiguous");
  });

  it("refuses to decide on a partial match", () => {
    expect(
      v("Issue #59 in theonion/fartscroll.js is open.", "I found theonion/fartscroll.js."),
    ).toBe("ambiguous");
  });

  it("refuses to decide a claim with no checkable atom", () => {
    const s = screenClaim("The response is helpful.", "t#0", "here you go");
    expect(s.verdict).toBe("ambiguous");
    expect(s.reason).toContain("no checkable atom");
  });

  it("tolerates currency and unit formatting differences", () => {
    expect(
      v("Game Title 23 generated $296.16 in sales.", "Game Title 23 made 296.16 dollars."),
    ).toBe("supported");
  });

  it("does not match a number inside a longer number", () => {
    // "5" must not match inside "2015"
    const s = screenClaim("There are 5 items and a Widget.", "t#0", "In 2015 the Widget shipped.");
    expect(s.missing).toContain("5");
  });

  it("matches a number that ends a sentence", () => {
    // Claims and answers both routinely end "...in 2013." — treating the period
    // as part of the number made every such claim a false negative.
    expect(v("The repo was created in 2013.", "It was created in 2013.")).toBe("supported");
  });

  it("does not match a whole number against its decimal extension", () => {
    const s = screenClaim("It made 296 dollars.", "t#0", "It made 296.16 dollars.");
    expect(s.missing).toContain("296");
  });

  it("records why it decided, for audit without a rerun", () => {
    const s = screenClaim("The repo was created in 2013.", "t#0", "created in 2013");
    expect(s.reason).toMatch(/atom/);
    expect(s.matched).toContain("2013");
  });
});

describe("screenClaims", () => {
  it("reports which claims still need a model and the auto rate", () => {
    const r = screenClaims(
      [
        "The repo was created in 2013.", // supported
        "The tone is friendly.", // no atom -> ambiguous
      ],
      "task1",
      "It was created in 2013.",
    );
    expect(r.screens.map((s) => s.verdict)).toEqual(["supported", "ambiguous"]);
    expect(r.ambiguous).toEqual([1]);
    expect(r.auto_rate).toBe(0.5);
    expect(r.screens[0].claim_id).toBe("task1#0");
  });
});

describe("self-consistency — a claim's own atoms must match the claim itself", () => {
  // Every divergence here was a real extraction bug: all-digit runs read as
  // commit hashes, semver split into fragments, ordinals ("21st") and keyed ids
  // ("CUST00010") blocked by a word boundary, and quoted URLs truncated because
  // a hash inside them was carved out first. Verified 0/145 on the real corpus.
  const CASES = [
    "The commit is 266f49303621cf01979ed65a02e1f92c1da09460.",
    "The average score is 101.05882352941177 across the file.",
    "It updated modelfusion from version 0.129.0 to 0.130.0.",
    "The commit completed on January 21st touched 3 files.",
    'The 10th customer has the Customer ID "CUST00010".',
    'The URL is "https://github.com/Atamyrat2005/snake-game/commit/abc1234".',
    "Game Title 23 generated a total of $296.16 in sales.",
    "The percentage change was -7.4%.",
    "It uses `@radix-ui/react-aspect-ratio` in the AspectRatio component.",
  ];

  for (const claim of CASES) {
    it(`matches itself: ${claim.slice(0, 52)}…`, () => {
      const normalized = normalizeText(claim);
      const missing = extractAtoms(claim).filter((a) => !atomPresent(a, normalized));
      expect(missing.map((a) => a.text)).toEqual([]);
    });
  }

  it("an all-digit run is a number, not a commit hash", () => {
    const atoms = extractAtoms("the average is 101.05882352941177");
    expect(atoms.some((a) => a.kind === "hash")).toBe(false);
    expect(atoms.some((a) => a.kind === "number" && a.text.startsWith("101."))).toBe(true);
  });

  it("a semver string stays whole", () => {
    const atoms = extractAtoms("from 0.129.0 to 0.130.0");
    expect(atoms.map((a) => a.text).sort()).toEqual(["0.129.0", "0.130.0"]);
  });
});
