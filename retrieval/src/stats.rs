//! Descriptive statistics for aggregating retrieval metrics across scenarios.

/// Mean, median, and population standard deviation of a sample.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct Stats {
    pub mean: f64,
    pub median: f64,
    pub stddev: f64,
}

pub fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

pub fn median(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut sorted = xs.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("non-NaN metric values"));
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

/// Population standard deviation (divides by `n`, not `n - 1`) since callers
/// describe the full evaluated set rather than a sample of a larger population.
pub fn stddev(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let m = mean(xs);
    let variance = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64;
    variance.sqrt()
}

pub fn summarize(xs: &[f64]) -> Stats {
    Stats {
        mean: mean(xs),
        median: median(xs),
        stddev: stddev(xs),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_slice_yields_zeros() {
        let s = summarize(&[]);
        assert_eq!(
            s,
            Stats {
                mean: 0.0,
                median: 0.0,
                stddev: 0.0
            }
        );
    }

    #[test]
    fn mean_of_known_values() {
        assert_eq!(mean(&[1.0, 2.0, 3.0, 4.0]), 2.5);
    }

    #[test]
    fn median_odd_length() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
    }

    #[test]
    fn median_even_length_averages_middle_two() {
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 2.5);
    }

    #[test]
    fn stddev_of_constant_is_zero() {
        assert_eq!(stddev(&[5.0, 5.0, 5.0]), 0.0);
    }

    #[test]
    fn stddev_matches_known_population_value() {
        // Population stddev of [2, 4, 4, 4, 5, 5, 7, 9] is 2.0 (textbook example).
        let xs = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!((stddev(&xs) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn summarize_bundles_all_three() {
        let s = summarize(&[1.0, 2.0, 3.0]);
        assert_eq!(s.mean, 2.0);
        assert_eq!(s.median, 2.0);
        assert!((s.stddev - (2.0_f64 / 3.0).sqrt()).abs() < 1e-9);
    }
}
