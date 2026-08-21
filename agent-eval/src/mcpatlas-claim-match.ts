// Deterministic claim screening.
//
// MCP-Atlas claims are overwhelmingly factual atoms rather than prose: of the 145
// claims in the coding task set, 74% contain a number, 98% contain a code
// identifier, and exactly one contains neither. That makes most of them checkable
// without a model.
//
// This module does NOT try to replace the judge. It decides only the cases where
// a wrong answer is implausible, and routes everything else onward. Four things
// stop it from being authoritative on its own:
//   - negation      "the repo was NOT created in 2013" contains 2013
//   - formatting    $296.16 / 296.16 / "296 dollars"; 7 / "seven"
//   - incidence     the answer mentions 2013 for an unrelated reason
//   - semantics     "balldontlie-mcp is a sports-related repository"
// So confident verdicts are gated on atom STRENGTH, not just presence.

export type AtomKind = "hash" | "quoted" | "code" | "identifier" | "number";

/** How much weight a single atom can carry on its own.
 *  strong  — near-unique in context (a commit hash, a quoted title, `a/b`)
 *  medium  — distinctive but not unique (a decimal, a 3+ digit number)
 *  weak    — could appear by coincidence (a small integer) */
export type AtomStrength = "strong" | "medium" | "weak";

export interface ClaimAtom {
  kind: AtomKind;
  strength: AtomStrength;
  /** As written in the claim. */
  text: string;
  /** Normalized forms; a match on ANY of them counts as present. */
  variants: string[];
}

const NEGATION_CUES = [
  "not ",
  "n't",
  "no ",
  "never",
  "none",
  "cannot",
  "unable",
  "does not",
  "did not",
  "isn't",
  "wasn't",
  "doesn't",
  "didn't",
];

const NUMBER_WORDS: Record<string, string> = {
  zero: "0",
  one: "1",
  two: "2",
  three: "3",
  four: "4",
  five: "5",
  six: "6",
  seven: "7",
  eight: "8",
  nine: "9",
  ten: "10",
  eleven: "11",
  twelve: "12",
};

/** Lowercase, collapse whitespace, and expand small number words so "seven"
 *  and "7" compare equal in both directions. */
export function normalizeText(s: string): string {
  let out = ` ${s.toLowerCase()} `.replace(/\s+/g, " ");
  for (const [word, digit] of Object.entries(NUMBER_WORDS)) {
    out = out.replace(new RegExp(`\\b${word}\\b`, "g"), digit);
  }
  return out;
}

/** Strip currency symbols, thousands separators and trailing zeros so
 *  "$296.16", "296.16" and "296.160" all reduce to the same key. */
export function normalizeNumber(raw: string): string {
  const cleaned = raw.replace(/[$€£,\s]/g, "");
  const n = Number(cleaned);
  if (!Number.isFinite(n)) return cleaned.toLowerCase();
  // Keep the sign; drop insignificant trailing zeros.
  return String(n);
}

function push(atoms: ClaimAtom[], atom: ClaimAtom): void {
  if (!atoms.some((a) => a.text === atom.text)) atoms.push(atom);
}

/**
 * Pull the checkable atoms out of a claim.
 *
 * Order matters: hashes and quoted spans are extracted before the generic
 * number/identifier passes so a commit hash is not shredded into digits.
 */
export function extractAtoms(claim: string): ClaimAtom[] {
  const atoms: ClaimAtom[] = [];
  let rest = claim;

  const consume = (re: RegExp, make: (m: RegExpExecArray) => ClaimAtom | null): void => {
    rest = rest.replace(re, (...args) => {
      const m = args as unknown as RegExpExecArray;
      const atom = make(m);
      if (atom) push(atoms, atom);
      return " ";
    });
  };

  // Explicitly delimited spans are extracted FIRST. A hash or number sitting
  // inside a quoted URL or a backticked span belongs to that span; carving it
  // out first leaves the quote truncated and unmatchable.
  // Quoted titles and phrases.
  consume(/["“]([^"”]{2,})["”]|'([^']{4,})'/g, (m) => {
    const t = m[1] ?? m[2];
    return t ? { kind: "quoted", strength: "strong", text: t, variants: [t.toLowerCase()] } : null;
  });

  // Backticked code spans.
  consume(/`([^`]+)`/g, (m) => ({
    kind: "code",
    strength: "strong",
    text: m[1],
    variants: [m[1].toLowerCase()],
  }));

  // Commit hashes / long hex ids. Requires at least one a-f letter: an all-digit
  // run is a NUMBER, not a hash, and treating e.g. the fractional digits of
  // 101.05882352941177 as a hash silently shreds the number that follows.
  // Matched by prefix later, because an answer may abbreviate 266f4930... to 266f493.
  consume(/\b(?=[0-9a-f]{7,40}\b)[0-9]*[a-f][0-9a-f]*\b/gi, (m) => ({
    kind: "hash",
    strength: "strong",
    text: m[0],
    variants: [m[0].toLowerCase()],
  }));

  // Scoped packages, owner/repo paths, dotted or hyphenated identifiers.
  consume(/@[\w.-]+\/[\w.-]+|\b[\w.-]+\/[\w.-]+\b/g, (m) => ({
    kind: "identifier",
    strength: "strong",
    text: m[0],
    variants: [m[0].toLowerCase()],
  }));

  // CamelCase / PascalCase names. Requires a real hump — a lowercase or digit
  // immediately followed by an uppercase — so that a sentence-initial "The" or
  // "There" is not mistaken for an identifier. Without that guard every claim
  // carries a spurious strong atom that any prose answer satisfies.
  consume(/\b[A-Za-z0-9]*[a-z0-9][A-Z][A-Za-z0-9]*\b/g, (m) => ({
    kind: "identifier",
    strength: "strong",
    text: m[0],
    variants: [m[0].toLowerCase()],
  }));

  // Hyphenated slugs like balldontlie-mcp. Deliberately MEDIUM, not strong:
  // ordinary English compounds ("sports-related") match this shape too, so a
  // slug may support a claim but must never be the sole basis for calling one
  // unsupported.
  consume(/\b[a-z0-9]+(?:-[a-z0-9]+){1,}\b/g, (m) => ({
    kind: "identifier",
    strength: "medium",
    text: m[0],
    variants: [m[0].toLowerCase()],
  }));

  // Dotted version strings (semver). Extracted before the number pass, which
  // would otherwise split 0.129.0 into "0.129" plus a stray "0".
  consume(/\bv?\d+\.\d+\.\d+(?:[-+][\w.]+)?\b/g, (m) => ({
    kind: "identifier",
    strength: "strong",
    text: m[0],
    variants: [m[0].toLowerCase(), m[0].toLowerCase().replace(/^v/, "")],
  }));

  // Numbers, including currency and percentages.
  consume(/-?\$?\d[\d,]*(?:\.\d+)?%?/g, (m) => {
    const raw = m[0];
    const bare = raw.replace(/[%$]/g, "");
    const norm = normalizeNumber(bare);
    const digits = norm.replace(/[^0-9]/g, "").length;
    const strength: AtomStrength =
      digits >= 3 || norm.includes(".") || raw.includes("%") || raw.includes("$")
        ? "medium"
        : "weak";
    return {
      kind: "number",
      strength,
      text: raw,
      variants: [...new Set([norm, bare.toLowerCase()])],
    };
  });

  return atoms;
}

/** Is this atom present in the answer? */
export function atomPresent(atom: ClaimAtom, normalizedAnswer: string): boolean {
  if (atom.kind === "hash") {
    // Prefix match: an answer may abbreviate a 40-char hash.
    const full = atom.variants[0];
    for (let len = full.length; len >= 7; len--) {
      if (normalizedAnswer.includes(full.slice(0, len))) return true;
    }
    return false;
  }
  if (atom.kind === "number") {
    // Guard against adjacent DIGITS, not adjacent letters. Blocking on digits
    // stops "5" matching inside "2015" and "296" matching inside "296.16".
    // Blocking on letters too would break the shapes these claims actually use:
    // ordinals ("21st", "10th"), versions ("v0.129") and keyed ids ("CUST00010").
    return atom.variants.some((v) => {
      const escaped = v.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      return new RegExp(`(?<![\\d.])${escaped}(?!\\.?\\d)`).test(normalizedAnswer);
    });
  }
  return atom.variants.some((v) => normalizedAnswer.includes(v));
}

export type ScreenVerdict = "supported" | "unsupported" | "ambiguous";

export interface ClaimScreen {
  claim_id: string;
  claim: string;
  atoms: ClaimAtom[];
  matched: string[];
  missing: string[];
  /** matched / total atoms. */
  atom_coverage: number;
  verdict: ScreenVerdict;
  /** Why the screener decided what it did — surfaced in the per-task report so a
   *  reviewer can audit an auto-scored claim without rerunning anything. */
  reason: string;
  /** A negation cue sits near a matched atom, so presence may mean the opposite. */
  negation_risk: boolean;
}

function hasNegationNear(normalizedAnswer: string, atom: ClaimAtom, window = 90): boolean {
  for (const v of atom.variants) {
    let idx = normalizedAnswer.indexOf(v);
    while (idx !== -1) {
      const start = Math.max(0, idx - window);
      const slice = normalizedAnswer.slice(start, idx + v.length);
      if (NEGATION_CUES.some((c) => slice.includes(c))) return true;
      idx = normalizedAnswer.indexOf(v, idx + 1);
    }
  }
  return false;
}

/**
 * Screen one claim.
 *
 * Auto-decides only where a wrong call is implausible:
 *   supported   — every atom present, at least one is not weak, no negation cue
 *   unsupported — no atom present, and at least one atom was strong (absence of a
 *                 weak atom such as a bare "5" says nothing)
 * Everything else is `ambiguous` and goes to the LLM judge.
 */
export function screenClaim(claim: string, claimId: string, answer: string): ClaimScreen {
  const atoms = extractAtoms(claim);
  const normalized = normalizeText(answer);
  const matched: string[] = [];
  const missing: string[] = [];
  for (const a of atoms) {
    (atomPresent(a, normalized) ? matched : missing).push(a.text);
  }
  const coverage = atoms.length ? matched.length / atoms.length : 0;
  const matchedAtoms = atoms.filter((a) => matched.includes(a.text));
  const negation_risk = matchedAtoms.some((a) => hasNegationNear(normalized, a));

  let verdict: ScreenVerdict = "ambiguous";
  let reason: string;

  if (!atoms.length) {
    reason = "no checkable atom in the claim";
  } else if (coverage === 1 && negation_risk) {
    reason = "all atoms present but a negation cue sits nearby";
  } else if (coverage === 1 && atoms.some((a) => a.strength !== "weak")) {
    verdict = "supported";
    reason = `all ${atoms.length} atom(s) present, incl. a non-weak atom`;
  } else if (coverage === 1) {
    reason = "all atoms present but all are weak — could be coincidence";
  } else if (coverage === 0 && atoms.some((a) => a.strength === "strong")) {
    verdict = "unsupported";
    reason = "no atom present, incl. a strong atom that should have appeared";
  } else if (coverage === 0) {
    reason = "no atom present but none were strong — absence is uninformative";
  } else {
    reason = `partial match (${matched.length}/${atoms.length} atoms)`;
  }

  return {
    claim_id: claimId,
    claim,
    atoms,
    matched,
    missing,
    atom_coverage: coverage,
    verdict,
    reason,
    negation_risk,
  };
}

export interface ScreenResult {
  screens: ClaimScreen[];
  /** Indices into `screens` that need the LLM. */
  ambiguous: number[];
  /** Share of claims the screener decided on its own. */
  auto_rate: number;
}

export function screenClaims(
  claims: readonly string[],
  taskId: string,
  answer: string,
): ScreenResult {
  const screens = claims.map((c, i) => screenClaim(c, `${taskId}#${i}`, answer));
  const ambiguous = screens
    .map((s, i) => (s.verdict === "ambiguous" ? i : -1))
    .filter((i) => i >= 0);
  return {
    screens,
    ambiguous,
    auto_rate: screens.length ? (screens.length - ambiguous.length) / screens.length : 0,
  };
}
