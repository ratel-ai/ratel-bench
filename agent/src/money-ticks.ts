/** Decimal USD-per-million rates become exact rational 10^-10 USD ticks. */
export function ratedTicks(tokens: number, rate: number): bigint | null {
  if (!Number.isSafeInteger(tokens) || tokens < 0 || !Number.isFinite(rate) || rate < 0)
    return null;
  const match = /^(\d+)(?:\.(\d+))?(?:e([+-]?\d+))?$/i.exec(rate.toString());
  if (!match) return null;
  const digits = BigInt(`${match[1]}${match[2] ?? ""}`);
  const scale = (match[2]?.length ?? 0) - Number(match[3] ?? 0);
  const numerator = BigInt(tokens) * digits * 10_000n * (scale < 0 ? 10n ** BigInt(-scale) : 1n);
  const denominator = scale > 0 ? 10n ** BigInt(scale) : 1n;
  return (numerator + denominator - 1n) / denominator;
}
