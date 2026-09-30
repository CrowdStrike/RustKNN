//! k-Nearest Neighbor Search State Management
//!
//! This module provides the data structures for tracking k-nearest neighbors during tree traversal.
//! The main structure is `KnnState`, which maintains the k best candidates found so far.
//!
//! # k=1 Specialization
//!
//! For k=1, `KnnState` uses a single `(f64, Option<&T>)` pair instead of a sorted array.
//! This reduces `kth_distance` to one field read and `insert` to one comparison + one
//! assignment.
//!
//! # k>1 Path: Sorted Flat Array
//!
//! For k>1, a sorted `Vec` of (distance, point) pairs is maintained in ascending order.
//! The worst (kth) candidate is always at position `len()-1`, enabling O(1) threshold
//! access. Insertion uses binary search + shift, which is O(k) per insert but operates
//! on a contiguous cache line.
//!
//! This replaces the previous BinaryHeap implementation, which had O(log k) insertion
//! but worse cache locality and higher constant factors from heap property maintenance
//! and pointer indirection for small k (the common k=5,10,25,50 range).

/// Internal backing store — specialized for k=1, sorted array for k>1.
enum KnnBacking<'a, T> {
    /// k=1 fast path: single best candidate tracked as (distance, point).
    #[cfg_attr(feature = "no-k1-special", allow(dead_code))]
    Single {
        best_dist: f64,
        best_point: Option<&'a T>,
    },
    /// k>1: sorted Vec of candidates in ascending distance order.
    /// Worst candidate is always at `entries.last()`. Capacity is pre-allocated to k.
    Sorted {
        entries: Vec<(f64, &'a T)>,
        k: usize,
    },
}

/// State for tracking k-nearest neighbors during search.
///
/// For k=1, uses a single-pair fast path (one comparison per insert).
/// For k>1, uses a sorted flat array (O(k) insert with excellent cache locality,
/// O(1) threshold peek).
///
/// # Deduplication Strategy
///
/// This implementation does NOT perform explicit deduplication by value comparison.
/// Instead, it relies on the `is_duplicate` flag in tree nodes to skip algorithm-created
/// duplicates during traversal.
///
/// # Type Parameters
///
/// * `'a` - Lifetime of references to points in the tree
/// * `T` - Type of points being searched
pub(crate) struct KnnState<'a, T> {
    backing: KnnBacking<'a, T>,
}

impl<'a, T> KnnState<'a, T> {
    /// Create a new k-NN state for tracking up to k neighbors.
    pub fn new(k: usize) -> Self {
        #[cfg(not(feature = "no-k1-special"))]
        if k == 1 {
            return KnnState {
                backing: KnnBacking::Single {
                    best_dist: f64::INFINITY,
                    best_point: None,
                },
            };
        }
        KnnState {
            backing: KnnBacking::Sorted {
                entries: Vec::with_capacity(k),
                k,
            },
        }
    }

    /// Get the kth-worst distance (the distance threshold for accepting new candidates).
    ///
    /// Returns `f64::INFINITY` if we haven't found k candidates yet (accept everything),
    /// otherwise returns the distance of the worst current candidate.
    #[inline]
    pub fn kth_distance(&self) -> f64 {
        match &self.backing {
            KnnBacking::Single { best_dist, .. } => *best_dist,
            KnnBacking::Sorted { entries, k } => {
                if entries.len() < *k {
                    f64::INFINITY
                } else {
                    // Worst is at the end (sorted ascending)
                    entries.last().unwrap().0
                }
            }
        }
    }

    /// Insert a candidate point if it's better than the current kth-worst.
    ///
    /// Returns `true` if the kth-distance changed (candidate accepted and worst
    /// distance now different). Lets callers skip a separate `kth_distance()` call.
    #[inline]
    pub fn insert(&mut self, point: &'a T, distance: f64) -> bool {
        match &mut self.backing {
            KnnBacking::Single { best_dist, best_point } => {
                if distance < *best_dist {
                    *best_dist = distance;
                    *best_point = Some(point);
                    true
                } else {
                    false
                }
            }
            KnnBacking::Sorted { entries, k } => {
                if entries.len() < *k {
                    // Not full yet: insert in sorted position
                    let pos = entries.partition_point(|e| e.0 <= distance);
                    entries.insert(pos, (distance, point));
                    // kth_distance changed from INFINITY to a real value when we fill up
                    entries.len() == *k
                } else if distance < entries.last().unwrap().0 {
                    // Better than current worst: binary search for insertion position,
                    // insert there, and pop the last (worst) entry.
                    // This is O(k) due to the shift, but operates on contiguous memory.
                    let pos = entries.partition_point(|e| e.0 <= distance);
                    // Pop the worst before inserting to avoid growing beyond capacity
                    entries.pop();
                    entries.insert(pos, (distance, point));
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Convert the state into a sorted vector of (point, distance) pairs.
    ///
    /// Consumes the state and returns all candidates sorted by distance (closest first).
    pub fn into_sorted_vec(self) -> Vec<(&'a T, f64)> {
        match self.backing {
            KnnBacking::Single { best_dist, best_point } => {
                match best_point {
                    Some(p) => vec![(p, best_dist)],
                    None => vec![],
                }
            }
            KnnBacking::Sorted { entries, .. } => {
                // Already sorted ascending by distance
                entries.into_iter().map(|(d, p)| (p, d)).collect()
            }
        }
    }

    /// Get the current number of candidates.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        match &self.backing {
            KnnBacking::Single { best_point, .. } => {
                if best_point.is_some() { 1 } else { 0 }
            }
            KnnBacking::Sorted { entries, .. } => entries.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knn_state_insert_under_k() {
        let mut state: KnnState<i32> = KnnState::new(3);
        let p1 = 10;
        let p2 = 20;
        let p3 = 30;

        state.insert(&p1, 5.0);
        state.insert(&p2, 3.0);
        assert_eq!(state.len(), 2);
        assert_eq!(state.kth_distance(), f64::INFINITY); // Not full yet

        state.insert(&p3, 7.0);
        assert_eq!(state.len(), 3);
        assert_eq!(state.kth_distance(), 7.0); // Worst is 7.0
    }

    #[test]
    fn test_knn_state_replace_worst() {
        let mut state: KnnState<i32> = KnnState::new(3);
        let p1 = 10;
        let p2 = 20;
        let p3 = 30;
        let p4 = 40;

        // Fill with 3 candidates
        state.insert(&p1, 5.0);
        state.insert(&p2, 3.0);
        state.insert(&p3, 7.0);
        assert_eq!(state.kth_distance(), 7.0);

        // Insert better candidate (should replace 7.0)
        state.insert(&p4, 4.0);
        assert_eq!(state.len(), 3);
        assert_eq!(state.kth_distance(), 5.0); // New worst is 5.0
    }

    #[test]
    fn test_knn_state_ignore_worse() {
        let mut state: KnnState<i32> = KnnState::new(2);
        let p1 = 10;
        let p2 = 20;
        let p3 = 30;

        state.insert(&p1, 3.0);
        state.insert(&p2, 5.0);
        assert_eq!(state.kth_distance(), 5.0);

        // Try to insert worse candidate (should be ignored)
        state.insert(&p3, 10.0);
        assert_eq!(state.len(), 2);
        assert_eq!(state.kth_distance(), 5.0); // Unchanged
    }

    #[test]
    fn test_knn_state_into_sorted_vec() {
        let mut state: KnnState<i32> = KnnState::new(5);
        let p1 = 10;
        let p2 = 20;
        let p3 = 30;

        state.insert(&p1, 5.0);
        state.insert(&p2, 2.0);
        state.insert(&p3, 8.0);

        let results = state.into_sorted_vec();
        assert_eq!(results.len(), 3);

        // Should be sorted by distance
        assert_eq!(results[0].1, 2.0);
        assert_eq!(results[1].1, 5.0);
        assert_eq!(results[2].1, 8.0);

        // Points should match
        assert_eq!(*results[0].0, 20);
        assert_eq!(*results[1].0, 10);
        assert_eq!(*results[2].0, 30);
    }

    #[test]
    fn test_knn_state_k_zero() {
        let state: KnnState<i32> = KnnState::new(0);
        assert_eq!(state.len(), 0);
        let results = state.into_sorted_vec();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_knn_state_k_one() {
        let mut state: KnnState<i32> = KnnState::new(1);
        let p1 = 10;
        let p2 = 20;

        state.insert(&p1, 5.0);
        assert_eq!(state.kth_distance(), 5.0);

        state.insert(&p2, 3.0); // Better
        assert_eq!(state.len(), 1);
        assert_eq!(state.kth_distance(), 3.0);

        let results = state.into_sorted_vec();
        assert_eq!(results.len(), 1);
        assert_eq!(*results[0].0, 20);
        assert_eq!(results[0].1, 3.0);
    }

    #[test]
    fn test_knn_state_sorted_order_maintained() {
        // Insert in random order, verify always sorted
        let mut state: KnnState<i32> = KnnState::new(5);
        let points = [10, 20, 30, 40, 50];
        let distances = [7.0, 2.0, 9.0, 1.0, 5.0];

        for i in 0..5 {
            state.insert(&points[i], distances[i]);
        }

        let results = state.into_sorted_vec();
        for i in 1..results.len() {
            assert!(
                results[i - 1].1 <= results[i].1,
                "Results not sorted at index {}: {} > {}",
                i,
                results[i - 1].1,
                results[i].1
            );
        }
    }

    #[test]
    fn test_insert_returns_changed() {
        let mut state: KnnState<i32> = KnnState::new(3);
        let p1 = 10;
        let p2 = 20;
        let p3 = 30;
        let p4 = 40;
        let p5 = 50;

        assert!(!state.insert(&p1, 5.0)); // len=1, kth still INFINITY
        assert!(!state.insert(&p2, 3.0)); // len=2, kth still INFINITY
        assert!(state.insert(&p3, 7.0)); // len=3 == k, kth transitions to 7.0

        assert!(state.insert(&p4, 4.0)); // displaces 7.0, new kth = 5.0
        assert!(!state.insert(&p5, 10.0)); // 10.0 >= 5.0, ignored
    }

    #[test]
    fn test_k_one_insert_returns_changed() {
        let mut state: KnnState<i32> = KnnState::new(1);
        let p1 = 10;
        let p2 = 20;
        let p3 = 30;

        // First insert: always accepted, kth changes from INFINITY to 5.0
        assert!(state.insert(&p1, 5.0));
        assert_eq!(state.kth_distance(), 5.0);

        // Better: accepted
        assert!(state.insert(&p2, 3.0));
        assert_eq!(state.kth_distance(), 3.0);

        // Worse: rejected
        assert!(!state.insert(&p3, 10.0));
        assert_eq!(state.kth_distance(), 3.0);
    }

    #[test]
    fn test_k_one_equal_distance_not_accepted() {
        let mut state: KnnState<i32> = KnnState::new(1);
        let p1 = 10;
        let p2 = 20;

        state.insert(&p1, 5.0);
        // Equal distance is NOT strictly less, so should be rejected
        assert!(!state.insert(&p2, 5.0));
        assert_eq!(state.kth_distance(), 5.0);
        // Point should still be p1
        let results = state.into_sorted_vec();
        assert_eq!(*results[0].0, 10);
    }
}
