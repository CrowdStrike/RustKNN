//! Multi-tree merging using binary reduction
//!
//! This module implements efficient merging of multiple SimplifiedCoverTrees
//! using a binary tree reduction strategy.

use crate::simplified::SimplifiedCoverTree;
use crate::Distance;
use rayon::prelude::*;

/// Merge multiple SimplifiedCoverTrees using binary tree reduction
///
/// This function takes a vector of trees and merges them in O(log n) rounds,
/// where n is the number of trees. Each round merges pairs of trees in parallel.
///
/// # Algorithm
///
/// ```text
/// Round 1: [T1, T2, T3, T4, T5, T6, T7, T8]
///          merge pairs in parallel
///          ↓
/// Round 2: [T12, T34, T56, T78]
///          merge pairs in parallel
///          ↓
/// Round 3: [T1234, T5678]
///          merge pairs in parallel
///          ↓
/// Result:  [T12345678]
/// ```
///
/// # Arguments
///
/// * `trees` - Vector of trees to merge (must be non-empty)
///
/// # Returns
///
/// A single merged tree containing all points from all input trees
///
/// # Panics
///
/// Panics if trees vector is empty
pub fn merge_all<T, D>(mut trees: Vec<SimplifiedCoverTree<T, D>>) -> SimplifiedCoverTree<T, D>
where
    T: Clone + Send + Sync,
    D: Distance<T> + Send + Sync + Clone,
{
    assert!(!trees.is_empty(), "Cannot merge empty vector of trees");

    // If only one tree, return it
    if trees.len() == 1 {
        return trees.into_iter().next().unwrap();
    }

    // Binary reduction: merge pairs until one tree remains
    while trees.len() > 1 {
        trees = trees
            .into_par_iter()
            .chunks(2)
            .map(|chunk| {
                if chunk.len() == 2 {
                    // Merge pair (deterministic merge, no randomization)
                    let mut iter = chunk.into_iter();
                    let tree1 = iter.next().unwrap();
                    let tree2 = iter.next().unwrap();
                    tree1.merge(tree2)
                } else {
                    // Odd one out
                    chunk.into_iter().next().unwrap()
                }
            })
            .collect();
    }

    trees.into_iter().next().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct SimpleDistance;

    impl Distance<f64> for SimpleDistance {
        fn distance(&self, p: &f64, q: &f64) -> f64 {
            (p - q).abs()
        }
    }

    #[test]
    fn test_merge_all_single_tree() {
        let mut tree = SimplifiedCoverTree::new(SimpleDistance, 1.3);
        tree.insert(1.0);
        tree.insert(2.0);

        let trees = vec![tree];
        let result = merge_all(trees);

        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_merge_all_two_trees() {
        let mut tree1 = SimplifiedCoverTree::new(SimpleDistance, 1.3);
        tree1.insert(1.0);

        let mut tree2 = SimplifiedCoverTree::new(SimpleDistance, 1.3);
        tree2.insert(10.0);

        let trees = vec![tree1, tree2];
        let result = merge_all(trees);

        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_merge_all_four_trees() {
        let mut tree1 = SimplifiedCoverTree::new(SimpleDistance, 1.3);
        tree1.insert(1.0);

        let mut tree2 = SimplifiedCoverTree::new(SimpleDistance, 1.3);
        tree2.insert(10.0);

        let mut tree3 = SimplifiedCoverTree::new(SimpleDistance, 1.3);
        tree3.insert(20.0);

        let mut tree4 = SimplifiedCoverTree::new(SimpleDistance, 1.3);
        tree4.insert(30.0);

        let trees = vec![tree1, tree2, tree3, tree4];
        let result = merge_all(trees);

        assert_eq!(result.len(), 4);
        assert!(result.find_nearest(&1.0).is_some());
        assert!(result.find_nearest(&10.0).is_some());
        assert!(result.find_nearest(&20.0).is_some());
        assert!(result.find_nearest(&30.0).is_some());
    }

    #[test]
    fn test_merge_all_odd_number() {
        let mut trees = Vec::new();
        for i in 0..5 {
            let mut tree = SimplifiedCoverTree::new(SimpleDistance, 1.3);
            tree.insert(i as f64 * 10.0);
            trees.push(tree);
        }

        let result = merge_all(trees);
        assert_eq!(result.len(), 5);
    }

    #[test]
    #[should_panic(expected = "Cannot merge empty vector of trees")]
    fn test_merge_all_empty_panics() {
        let trees: Vec<SimplifiedCoverTree<f64, SimpleDistance>> = Vec::new();
        let _ = merge_all(trees);
    }
}
