//! Distance Trait Module
//!
//! This module defines the `Distance` trait, which provides an interface for computing
//! distances between points in arbitrary metric spaces.
//!
//! # Metric Space Requirements
//!
//! A valid metric must satisfy these mathematical properties:
//! 1. **Non-negativity**: d(p, q) ≥ 0 for all points p, q
//! 2. **Identity**: d(p, q) = 0 if and only if p = q
//! 3. **Symmetry**: d(p, q) = d(q, p) for all points p, q
//! 4. **Triangle inequality**: d(p, r) ≤ d(p, q) + d(q, r) for all points p, q, r
//!
//! # Examples
//!
//! ```
//! use rustknn::Distance;
//!
//! // Define a simple 2D point
//! #[derive(Clone)]
//! struct Point2D {
//!     x: f64,
//!     y: f64,
//! }
//!
//! // Define a distance metric (Euclidean distance)
//! struct EuclideanDistance;
//!
//! impl Distance<Point2D> for EuclideanDistance {
//!     fn distance(&self, p: &Point2D, q: &Point2D) -> f64 {
//!         let dx = p.x - q.x;
//!         let dy = p.y - q.y;
//!         (dx * dx + dy * dy).sqrt()
//!     }
//! }
//!
//! // Now you can use it
//! let metric = EuclideanDistance;
//! let p1 = Point2D { x: 0.0, y: 0.0 };
//! let p2 = Point2D { x: 3.0, y: 4.0 };
//! let dist = metric.distance(&p1, &p2);
//! assert_eq!(dist, 5.0); // 3-4-5 triangle
//! ```

/// A trait for computing distances between points in a metric space.
///
/// This trait defines the interface for distance functions. Any type that implements
/// this trait can be used as a metric for the cover tree.
///
/// # Type Parameters
///
/// * `T` - The type of points in the metric space. This is a **generic type parameter**,
///   which means the Distance trait can work with any point type.
///
/// # Examples
///
/// ## Example 1: Euclidean Distance in 2D
///
/// ```
/// use rustknn::Distance;
///
/// #[derive(Clone)]
/// struct Point2D { x: f64, y: f64 }
///
/// struct EuclideanDistance;
///
/// impl Distance<Point2D> for EuclideanDistance {
///     fn distance(&self, p: &Point2D, q: &Point2D) -> f64 {
///         let dx = p.x - q.x;
///         let dy = p.y - q.y;
///         (dx * dx + dy * dy).sqrt()
///     }
/// }
/// ```
///
/// ## Example 2: Manhattan Distance (L1 norm)
///
/// ```
/// use rustknn::Distance;
///
/// #[derive(Clone)]
/// struct Point2D { x: f64, y: f64 }
///
/// struct ManhattanDistance;
///
/// impl Distance<Point2D> for ManhattanDistance {
///     fn distance(&self, p: &Point2D, q: &Point2D) -> f64 {
///         (p.x - q.x).abs() + (p.y - q.y).abs()
///     }
/// }
/// ```
///
/// ## Example 3: Discrete Metric (0 if equal, 1 if different)
///
/// ```
/// use rustknn::Distance;
///
/// struct DiscreteDistance;
///
/// impl Distance<i32> for DiscreteDistance {
///     fn distance(&self, p: &i32, q: &i32) -> f64 {
///         if p == q { 0.0 } else { 1.0 }
///     }
/// }
/// ```
pub trait Distance<T> {
    /// Computes the distance between two points.
    ///
    /// # Arguments
    ///
    /// * `p` - The first point (borrowed - we don't take ownership)
    /// * `q` - The second point (borrowed - we don't take ownership)
    ///
    /// # Returns
    ///
    /// The distance between `p` and `q` as a 64-bit floating point number.
    ///
    /// # Rust Ownership Note
    ///
    /// This method borrows `&self`, `p`, and `q`. After calling this method, you still
    /// own all three values and can continue using them. Nothing is consumed or moved.
    ///
    /// # Example
    ///
    /// ```
    /// use rustknn::Distance;
    ///
    /// # #[derive(Clone, Debug)]
    /// # struct Point2D { x: f64, y: f64 }
    /// # struct EuclideanDistance;
    /// # impl Distance<Point2D> for EuclideanDistance {
    /// #     fn distance(&self, p: &Point2D, q: &Point2D) -> f64 {
    /// #         let dx = p.x - q.x;
    /// #         let dy = p.y - q.y;
    /// #         (dx * dx + dy * dy).sqrt()
    /// #     }
    /// # }
    /// let metric = EuclideanDistance;
    /// let p1 = Point2D { x: 0.0, y: 0.0 };
    /// let p2 = Point2D { x: 3.0, y: 4.0 };
    ///
    /// // Call distance - note we use & to borrow
    /// let d = metric.distance(&p1, &p2);
    ///
    /// // We can still use metric, p1, and p2 here!
    /// let d2 = metric.distance(&p1, &p2); // Can call again
    /// println!("{:?}", p1); // Can still use p1
    /// ```
    fn distance(&self, p: &T, q: &T) -> f64;

    /// Computes the distance between two points, but may return early if the
    /// distance is guaranteed to exceed `upper_bound`.
    ///
    /// The returned value is:
    /// - The exact distance, if it is ≤ `upper_bound`
    /// - Some value > `upper_bound`, if the true distance exceeds the bound
    ///   (not necessarily the exact distance — partial computation is allowed)
    ///
    /// This enables early termination in high-dimensional Euclidean distance
    /// where partial sums can prove a point is too far before processing all
    /// dimensions.
    ///
    /// The default implementation simply calls `distance()` (no short-circuit).
    fn distance_with_bound(&self, p: &T, q: &T, upper_bound: f64) -> f64 {
        let _ = upper_bound;
        self.distance(p, q)
    }
}

// ---------------------------------------------------------------------------
// Distance computation counter (instrumentation)
// ---------------------------------------------------------------------------

#[cfg(feature = "instrument")]
use std::cell::Cell;

#[cfg(feature = "instrument")]
thread_local! {
    /// Per-thread distance computation counter. Incremented by every call to
    /// `EuclideanDistance::distance` or `EuclideanDistance::distance_with_bound`.
    /// Use `reset_distance_count()` before a benchmark and `get_distance_count()`
    /// after to measure the number of metric evaluations.
    static DISTANCE_COUNT: Cell<u64> = const { Cell::new(0) };
}

/// Reset the thread-local distance computation counter to zero.
#[cfg(feature = "instrument")]
pub fn reset_distance_count() {
    DISTANCE_COUNT.with(|c| c.set(0));
}

/// Reset the thread-local distance computation counter to zero.
/// (No-op when the `instrument` feature is disabled.)
#[cfg(not(feature = "instrument"))]
pub fn reset_distance_count() {}

/// Read the thread-local distance computation counter.
#[cfg(feature = "instrument")]
pub fn get_distance_count() -> u64 {
    DISTANCE_COUNT.with(|c| c.get())
}

/// Read the thread-local distance computation counter.
/// (Always returns 0 when the `instrument` feature is disabled.)
#[cfg(not(feature = "instrument"))]
pub fn get_distance_count() -> u64 {
    0
}

/// Increment the thread-local distance counter by 1.
#[cfg(feature = "instrument")]
#[inline(always)]
fn inc_distance_count() {
    DISTANCE_COUNT.with(|c| c.set(c.get() + 1));
}

/// Increment the thread-local distance counter by 1.
/// (No-op when the `instrument` feature is disabled.)
#[cfg(not(feature = "instrument"))]
#[inline(always)]
fn inc_distance_count() {}

// ---------------------------------------------------------------------------
// Built-in EuclideanDistance
// ---------------------------------------------------------------------------

/// Built-in Euclidean (L2) distance metric for `Vec<f64>` and `Vec<f32>`.
///
/// `distance_with_bound` checks partial sums periodically and returns early
/// when the accumulated squared difference already exceeds `upper_bound²`.
///
/// # Examples
///
/// ```
/// use rustknn::{Distance, EuclideanDistance};
///
/// let metric = EuclideanDistance;
/// let p = vec![0.0, 0.0, 3.0];
/// let q = vec![0.0, 4.0, 0.0];
/// assert!((metric.distance(&p, &q) - 5.0).abs() < 1e-10);
/// ```
#[derive(Clone, Debug)]
pub struct EuclideanDistance;

impl Distance<Vec<f64>> for EuclideanDistance {
    #[inline]
    fn distance(&self, p: &Vec<f64>, q: &Vec<f64>) -> f64 {
        inc_distance_count();
        assert_eq!(p.len(), q.len(), "EuclideanDistance: dimension mismatch");
        let n = p.len();
        let pp = p.as_ptr();
        let qp = q.as_ptr();
        // 4 independent accumulators processing 8 elements per iteration.
        // This saturates both FMA ports by breaking the serial dependency chain
        // that a single accumulator creates (~4-5 cycle latency per mul_add).
        // Uses raw pointers to eliminate bounds checks, enabling the compiler to
        // vectorize without inserting branch instructions per element.
        let mut s0 = 0.0_f64;
        let mut s1 = 0.0_f64;
        let mut s2 = 0.0_f64;
        let mut s3 = 0.0_f64;
        let mut i = 0;
        unsafe {
            while i + 7 < n {
                let d0 = *pp.add(i)     - *qp.add(i);
                let d1 = *pp.add(i + 1) - *qp.add(i + 1);
                let d2 = *pp.add(i + 2) - *qp.add(i + 2);
                let d3 = *pp.add(i + 3) - *qp.add(i + 3);
                let d4 = *pp.add(i + 4) - *qp.add(i + 4);
                let d5 = *pp.add(i + 5) - *qp.add(i + 5);
                let d6 = *pp.add(i + 6) - *qp.add(i + 6);
                let d7 = *pp.add(i + 7) - *qp.add(i + 7);
                s0 = d0.mul_add(d0, d1.mul_add(d1, s0));
                s1 = d2.mul_add(d2, d3.mul_add(d3, s1));
                s2 = d4.mul_add(d4, d5.mul_add(d5, s2));
                s3 = d6.mul_add(d6, d7.mul_add(d7, s3));
                i += 8;
            }
            // Remainder: 2-element pairs into s0
            while i + 1 < n {
                let d0 = *pp.add(i) - *qp.add(i);
                let d1 = *pp.add(i + 1) - *qp.add(i + 1);
                s0 = d0.mul_add(d0, d1.mul_add(d1, s0));
                i += 2;
            }
            // Final odd element
            if i < n {
                let d = *pp.add(i) - *qp.add(i);
                s0 = d.mul_add(d, s0);
            }
        }
        ((s0 + s1) + (s2 + s3)).sqrt()
    }

    #[inline]
    fn distance_with_bound(&self, p: &Vec<f64>, q: &Vec<f64>, upper_bound: f64) -> f64 {
        // When no-early-exit feature is enabled, skip the early termination logic
        // and always compute the full distance. Ablation switch.
        #[cfg(feature = "no-early-exit")]
        {
            let _ = upper_bound;
            return self.distance(p, q);
        }
        #[allow(unreachable_code)]
        {
        inc_distance_count();
        assert_eq!(p.len(), q.len(), "EuclideanDistance: dimension mismatch");
        let ub_sq = upper_bound * upper_bound;
        let n = p.len();
        let pp = p.as_ptr();
        let qp = q.as_ptr();
        let mut s0 = 0.0_f64;
        let mut s1 = 0.0_f64;
        let mut s2 = 0.0_f64;
        let mut s3 = 0.0_f64;
        let mut i = 0;
        unsafe {
            if n >= 64 {
                // High-dim path: check bound every 64 elements to enable early exit.
                // Each 64-element block = 8 iterations of the 8-element loop.
                while i + 63 < n {
                    for _ in 0..8 {
                        let d0 = *pp.add(i)     - *qp.add(i);
                        let d1 = *pp.add(i + 1) - *qp.add(i + 1);
                        let d2 = *pp.add(i + 2) - *qp.add(i + 2);
                        let d3 = *pp.add(i + 3) - *qp.add(i + 3);
                        let d4 = *pp.add(i + 4) - *qp.add(i + 4);
                        let d5 = *pp.add(i + 5) - *qp.add(i + 5);
                        let d6 = *pp.add(i + 6) - *qp.add(i + 6);
                        let d7 = *pp.add(i + 7) - *qp.add(i + 7);
                        s0 = d0.mul_add(d0, d1.mul_add(d1, s0));
                        s1 = d2.mul_add(d2, d3.mul_add(d3, s1));
                        s2 = d4.mul_add(d4, d5.mul_add(d5, s2));
                        s3 = d6.mul_add(d6, d7.mul_add(d7, s3));
                        i += 8;
                    }
                    if (s0 + s1) + (s2 + s3) > ub_sq {
                        // Early exit: distance exceeds bound. Return sentinel > upper_bound.
                        // All callers prune when distance > bound, so exact value is unneeded.
                        // Saves ~12-20 cycles per pruned distance computation (no sqrt).
                        return upper_bound + 1.0;
                    }
                }
            } else if n >= 16 {
                // Medium-dim path: check bound every 16 elements (2 x 8-element blocks).
                // For datasets with 16-63 dimensions (e.g. ionosphere=34, covtype=55),
                // the high-dim path never triggers — this enables early exit.
                while i + 15 < n {
                    for _ in 0..2 {
                        let d0 = *pp.add(i)     - *qp.add(i);
                        let d1 = *pp.add(i + 1) - *qp.add(i + 1);
                        let d2 = *pp.add(i + 2) - *qp.add(i + 2);
                        let d3 = *pp.add(i + 3) - *qp.add(i + 3);
                        let d4 = *pp.add(i + 4) - *qp.add(i + 4);
                        let d5 = *pp.add(i + 5) - *qp.add(i + 5);
                        let d6 = *pp.add(i + 6) - *qp.add(i + 6);
                        let d7 = *pp.add(i + 7) - *qp.add(i + 7);
                        s0 = d0.mul_add(d0, d1.mul_add(d1, s0));
                        s1 = d2.mul_add(d2, d3.mul_add(d3, s1));
                        s2 = d4.mul_add(d4, d5.mul_add(d5, s2));
                        s3 = d6.mul_add(d6, d7.mul_add(d7, s3));
                        i += 8;
                    }
                    if (s0 + s1) + (s2 + s3) > ub_sq {
                        return upper_bound + 1.0;
                    }
                }
            }
            // Remaining full 8-element blocks (no bound check — too few dims left)
            while i + 7 < n {
                let d0 = *pp.add(i)     - *qp.add(i);
                let d1 = *pp.add(i + 1) - *qp.add(i + 1);
                let d2 = *pp.add(i + 2) - *qp.add(i + 2);
                let d3 = *pp.add(i + 3) - *qp.add(i + 3);
                let d4 = *pp.add(i + 4) - *qp.add(i + 4);
                let d5 = *pp.add(i + 5) - *qp.add(i + 5);
                let d6 = *pp.add(i + 6) - *qp.add(i + 6);
                let d7 = *pp.add(i + 7) - *qp.add(i + 7);
                s0 = d0.mul_add(d0, d1.mul_add(d1, s0));
                s1 = d2.mul_add(d2, d3.mul_add(d3, s1));
                s2 = d4.mul_add(d4, d5.mul_add(d5, s2));
                s3 = d6.mul_add(d6, d7.mul_add(d7, s3));
                i += 8;
            }
            // Remainder pairs
            while i + 1 < n {
                let d0 = *pp.add(i) - *qp.add(i);
                let d1 = *pp.add(i + 1) - *qp.add(i + 1);
                s0 = d0.mul_add(d0, d1.mul_add(d1, s0));
                i += 2;
            }
            if i < n {
                let d = *pp.add(i) - *qp.add(i);
                s0 = d.mul_add(d, s0);
            }
        }
        ((s0 + s1) + (s2 + s3)).sqrt()
        } // #[allow(unreachable_code)]
    }
}

impl Distance<Vec<f32>> for EuclideanDistance {
    #[inline]
    fn distance(&self, p: &Vec<f32>, q: &Vec<f32>) -> f64 {
        inc_distance_count();
        assert_eq!(p.len(), q.len(), "EuclideanDistance: dimension mismatch");
        // Accumulate in f32: halves memory bandwidth and avoids per-element
        // as-f64 conversion
        let n = p.len();
        let pp = p.as_ptr();
        let qp = q.as_ptr();
        let mut s0 = 0.0_f32;
        let mut s1 = 0.0_f32;
        let mut s2 = 0.0_f32;
        let mut s3 = 0.0_f32;
        let mut i = 0;
        unsafe {
            while i + 7 < n {
                let d0 = *pp.add(i)     - *qp.add(i);
                let d1 = *pp.add(i + 1) - *qp.add(i + 1);
                let d2 = *pp.add(i + 2) - *qp.add(i + 2);
                let d3 = *pp.add(i + 3) - *qp.add(i + 3);
                let d4 = *pp.add(i + 4) - *qp.add(i + 4);
                let d5 = *pp.add(i + 5) - *qp.add(i + 5);
                let d6 = *pp.add(i + 6) - *qp.add(i + 6);
                let d7 = *pp.add(i + 7) - *qp.add(i + 7);
                s0 = d0.mul_add(d0, d1.mul_add(d1, s0));
                s1 = d2.mul_add(d2, d3.mul_add(d3, s1));
                s2 = d4.mul_add(d4, d5.mul_add(d5, s2));
                s3 = d6.mul_add(d6, d7.mul_add(d7, s3));
                i += 8;
            }
            while i + 1 < n {
                let d0 = *pp.add(i) - *qp.add(i);
                let d1 = *pp.add(i + 1) - *qp.add(i + 1);
                s0 = d0.mul_add(d0, d1.mul_add(d1, s0));
                i += 2;
            }
            if i < n {
                let d = *pp.add(i) - *qp.add(i);
                s0 = d.mul_add(d, s0);
            }
        }
        (((s0 + s1) + (s2 + s3)) as f64).sqrt()
    }

    #[inline]
    fn distance_with_bound(&self, p: &Vec<f32>, q: &Vec<f32>, upper_bound: f64) -> f64 {
        inc_distance_count();
        assert_eq!(p.len(), q.len(), "EuclideanDistance: dimension mismatch");
        let ub_sq = (upper_bound * upper_bound) as f32;
        let n = p.len();
        let pp = p.as_ptr();
        let qp = q.as_ptr();
        let mut s0 = 0.0_f32;
        let mut s1 = 0.0_f32;
        let mut s2 = 0.0_f32;
        let mut s3 = 0.0_f32;
        let mut i = 0;
        unsafe {
            if n >= 64 {
                // High-dim path: check bound every 64 elements
                while i + 63 < n {
                    for _ in 0..8 {
                        let d0 = *pp.add(i)     - *qp.add(i);
                        let d1 = *pp.add(i + 1) - *qp.add(i + 1);
                        let d2 = *pp.add(i + 2) - *qp.add(i + 2);
                        let d3 = *pp.add(i + 3) - *qp.add(i + 3);
                        let d4 = *pp.add(i + 4) - *qp.add(i + 4);
                        let d5 = *pp.add(i + 5) - *qp.add(i + 5);
                        let d6 = *pp.add(i + 6) - *qp.add(i + 6);
                        let d7 = *pp.add(i + 7) - *qp.add(i + 7);
                        s0 = d0.mul_add(d0, d1.mul_add(d1, s0));
                        s1 = d2.mul_add(d2, d3.mul_add(d3, s1));
                        s2 = d4.mul_add(d4, d5.mul_add(d5, s2));
                        s3 = d6.mul_add(d6, d7.mul_add(d7, s3));
                        i += 8;
                    }
                    if (s0 + s1) + (s2 + s3) > ub_sq {
                        return upper_bound + 1.0;
                    }
                }
            } else if n >= 16 {
                // Medium-dim path: check bound every 16 elements
                while i + 15 < n {
                    for _ in 0..2 {
                        let d0 = *pp.add(i)     - *qp.add(i);
                        let d1 = *pp.add(i + 1) - *qp.add(i + 1);
                        let d2 = *pp.add(i + 2) - *qp.add(i + 2);
                        let d3 = *pp.add(i + 3) - *qp.add(i + 3);
                        let d4 = *pp.add(i + 4) - *qp.add(i + 4);
                        let d5 = *pp.add(i + 5) - *qp.add(i + 5);
                        let d6 = *pp.add(i + 6) - *qp.add(i + 6);
                        let d7 = *pp.add(i + 7) - *qp.add(i + 7);
                        s0 = d0.mul_add(d0, d1.mul_add(d1, s0));
                        s1 = d2.mul_add(d2, d3.mul_add(d3, s1));
                        s2 = d4.mul_add(d4, d5.mul_add(d5, s2));
                        s3 = d6.mul_add(d6, d7.mul_add(d7, s3));
                        i += 8;
                    }
                    if (s0 + s1) + (s2 + s3) > ub_sq {
                        return upper_bound + 1.0;
                    }
                }
            }
            while i + 7 < n {
                let d0 = *pp.add(i)     - *qp.add(i);
                let d1 = *pp.add(i + 1) - *qp.add(i + 1);
                let d2 = *pp.add(i + 2) - *qp.add(i + 2);
                let d3 = *pp.add(i + 3) - *qp.add(i + 3);
                let d4 = *pp.add(i + 4) - *qp.add(i + 4);
                let d5 = *pp.add(i + 5) - *qp.add(i + 5);
                let d6 = *pp.add(i + 6) - *qp.add(i + 6);
                let d7 = *pp.add(i + 7) - *qp.add(i + 7);
                s0 = d0.mul_add(d0, d1.mul_add(d1, s0));
                s1 = d2.mul_add(d2, d3.mul_add(d3, s1));
                s2 = d4.mul_add(d4, d5.mul_add(d5, s2));
                s3 = d6.mul_add(d6, d7.mul_add(d7, s3));
                i += 8;
            }
            while i + 1 < n {
                let d0 = *pp.add(i) - *qp.add(i);
                let d1 = *pp.add(i + 1) - *qp.add(i + 1);
                s0 = d0.mul_add(d0, d1.mul_add(d1, s0));
                i += 2;
            }
            if i < n {
                let d = *pp.add(i) - *qp.add(i);
                s0 = d.mul_add(d, s0);
            }
        }
        (((s0 + s1) + (s2 + s3)) as f64).sqrt()
    }
}

// ---------------------------------------------------------------------------
// Built-in ManhattanDistance
// ---------------------------------------------------------------------------

/// Built-in Manhattan (L1) distance metric for `Vec<f64>`.
///
/// `distance_with_bound` accumulates `|p[i] - q[i]|` and returns early
/// when the partial sum already exceeds `upper_bound`.
///
/// # Examples
///
/// ```
/// use rustknn::{Distance, ManhattanDistance};
///
/// let metric = ManhattanDistance;
/// let p = vec![0.0, 0.0, 3.0];
/// let q = vec![0.0, 4.0, 0.0];
/// assert!((metric.distance(&p, &q) - 7.0).abs() < 1e-10);
/// ```
#[derive(Clone, Debug)]
pub struct ManhattanDistance;

impl Distance<Vec<f64>> for ManhattanDistance {
    #[inline]
    fn distance(&self, p: &Vec<f64>, q: &Vec<f64>) -> f64 {
        inc_distance_count();
        assert_eq!(p.len(), q.len(), "ManhattanDistance: dimension mismatch");
        let n = p.len();
        let pp = p.as_ptr();
        let qp = q.as_ptr();
        // 4 independent accumulators for instruction-level parallelism.
        // Same pattern as EuclideanDistance but accumulates absolute differences.
        let mut s0 = 0.0_f64;
        let mut s1 = 0.0_f64;
        let mut s2 = 0.0_f64;
        let mut s3 = 0.0_f64;
        let mut i = 0;
        unsafe {
            while i + 7 < n {
                s0 += (*pp.add(i)     - *qp.add(i)).abs();
                s1 += (*pp.add(i + 1) - *qp.add(i + 1)).abs();
                s2 += (*pp.add(i + 2) - *qp.add(i + 2)).abs();
                s3 += (*pp.add(i + 3) - *qp.add(i + 3)).abs();
                s0 += (*pp.add(i + 4) - *qp.add(i + 4)).abs();
                s1 += (*pp.add(i + 5) - *qp.add(i + 5)).abs();
                s2 += (*pp.add(i + 6) - *qp.add(i + 6)).abs();
                s3 += (*pp.add(i + 7) - *qp.add(i + 7)).abs();
                i += 8;
            }
            while i < n {
                s0 += (*pp.add(i) - *qp.add(i)).abs();
                i += 1;
            }
        }
        (s0 + s1) + (s2 + s3)
    }

    #[inline]
    fn distance_with_bound(&self, p: &Vec<f64>, q: &Vec<f64>, upper_bound: f64) -> f64 {
        #[cfg(feature = "no-early-exit")]
        {
            let _ = upper_bound;
            return self.distance(p, q);
        }
        #[allow(unreachable_code)]
        {
        inc_distance_count();
        assert_eq!(p.len(), q.len(), "ManhattanDistance: dimension mismatch");
        let n = p.len();
        let pp = p.as_ptr();
        let qp = q.as_ptr();
        let mut s0 = 0.0_f64;
        let mut s1 = 0.0_f64;
        let mut s2 = 0.0_f64;
        let mut s3 = 0.0_f64;
        let mut i = 0;
        unsafe {
            if n >= 32 {
                // Check bound every 32 elements
                while i + 31 < n {
                    for _ in 0..4 {
                        s0 += (*pp.add(i)     - *qp.add(i)).abs();
                        s1 += (*pp.add(i + 1) - *qp.add(i + 1)).abs();
                        s2 += (*pp.add(i + 2) - *qp.add(i + 2)).abs();
                        s3 += (*pp.add(i + 3) - *qp.add(i + 3)).abs();
                        s0 += (*pp.add(i + 4) - *qp.add(i + 4)).abs();
                        s1 += (*pp.add(i + 5) - *qp.add(i + 5)).abs();
                        s2 += (*pp.add(i + 6) - *qp.add(i + 6)).abs();
                        s3 += (*pp.add(i + 7) - *qp.add(i + 7)).abs();
                        i += 8;
                    }
                    if (s0 + s1) + (s2 + s3) > upper_bound {
                        return upper_bound + 1.0;
                    }
                }
            } else if n >= 16 {
                // Check bound every 16 elements
                while i + 15 < n {
                    for _ in 0..2 {
                        s0 += (*pp.add(i)     - *qp.add(i)).abs();
                        s1 += (*pp.add(i + 1) - *qp.add(i + 1)).abs();
                        s2 += (*pp.add(i + 2) - *qp.add(i + 2)).abs();
                        s3 += (*pp.add(i + 3) - *qp.add(i + 3)).abs();
                        s0 += (*pp.add(i + 4) - *qp.add(i + 4)).abs();
                        s1 += (*pp.add(i + 5) - *qp.add(i + 5)).abs();
                        s2 += (*pp.add(i + 6) - *qp.add(i + 6)).abs();
                        s3 += (*pp.add(i + 7) - *qp.add(i + 7)).abs();
                        i += 8;
                    }
                    if (s0 + s1) + (s2 + s3) > upper_bound {
                        return upper_bound + 1.0;
                    }
                }
            }
            // Remaining elements (no bound check - too few dims left)
            while i + 7 < n {
                s0 += (*pp.add(i)     - *qp.add(i)).abs();
                s1 += (*pp.add(i + 1) - *qp.add(i + 1)).abs();
                s2 += (*pp.add(i + 2) - *qp.add(i + 2)).abs();
                s3 += (*pp.add(i + 3) - *qp.add(i + 3)).abs();
                s0 += (*pp.add(i + 4) - *qp.add(i + 4)).abs();
                s1 += (*pp.add(i + 5) - *qp.add(i + 5)).abs();
                s2 += (*pp.add(i + 6) - *qp.add(i + 6)).abs();
                s3 += (*pp.add(i + 7) - *qp.add(i + 7)).abs();
                i += 8;
            }
            while i < n {
                s0 += (*pp.add(i) - *qp.add(i)).abs();
                i += 1;
            }
        }
        (s0 + s1) + (s2 + s3)
        } // #[allow(unreachable_code)]
    }
}
