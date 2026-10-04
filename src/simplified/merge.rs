//! Algorithm 4: Tree Merging for SimplifiedCoverTree
//!
//! This module implements the merge operation for SimplifiedCoverTree based on
//! "Faster Cover Trees" (ICML 2015) Algorithm 4.
//!
//! The merge algorithm combines two cover trees while maintaining all three invariants:
//! 1. Leveling: children at parent.level - 1
//! 2. Covering: distance(parent, child) ≤ base^parent.level
//! 3. Separating: distance(child1, child2) > base^(parent.level - 1)
//!
//! Key steps:
//! 1. Align tree levels (raise lower tree to match higher)
//! 2. Partition tree2's children into covered/not-covered by tree1
//! 3. Recursively merge covered children with tree1's children
//! 4. Insert tree2's root into tree1
//! 5. Handle uncovered children

use crate::node::Node;
use crate::Distance;
use std::mem;

/// Largest number of levels `raise_tree_level` will add when aligning trees in a merge.
///
/// Each level adds one placeholder node to a chain, and later passes recurse over that
/// chain. At base 1.3 two root levels can never differ by this much (every finite `f64`
/// distance maps to a level within about ±2,840), so the limit only applies to data
/// spanning hundreds of orders of magnitude at bases close to `MIN_BASE`.
const MAX_MERGE_LEVEL_GAP: i64 = 4096;

/// Implementation of merge operations for SimplifiedCoverTree
pub struct MergeImpl;

impl MergeImpl {
    /// Merge two cover tree roots, consuming both and returning the merged tree
    ///
    /// # Arguments
    /// * `tree1` - First tree root (will be the base tree)
    /// * `tree2` - Second tree root (will be merged into tree1)
    /// * `metric` - Distance metric
    /// * `base` - Base value for covdist/sepdist calculations
    ///
    /// # Returns
    /// Merged tree maintaining all three invariants
    ///
    /// NOTE: When aligning levels, intermediate nodes are created with cloned points
    /// (marked `is_duplicate = true`). These are automatically skipped during k-NN
    /// queries to avoid returning the same logical point multiple times.
    pub fn merge<T: Clone, D: Distance<T>>(
        tree1: Box<Node<T>>,
        tree2: Box<Node<T>>,
        metric: &D,
        base: f64,
    ) -> Box<Node<T>> {
        let _guard = crate::core::utils::StackGuard::enter();
        // Step 1: Align levels (raise lower tree to match higher)
        let (tree1, tree2) = Self::align_levels(tree1, tree2);

        // Step 2: Check if tree2.root is within covering distance of tree1.root
        let dist = metric.distance(&tree1.point, &tree2.point);

        if dist > tree1.covdist(base) {
            // tree1 cannot cover tree2, create new parent covering both
            let mut result = Self::create_common_parent(tree1, tree2, dist, base);
            // Recompute maxdist because create_common_parent uses triangle inequality (upper bound)
            result.recalc_maxdist_approx(metric);
            result
        } else {
            // tree2 is within covering distance, merge it into tree1
            Self::merge_into(tree1, tree2, metric, base)
        }
    }

    /// Align the levels of two trees by raising the lower one
    ///
    /// After alignment, both trees will be at the same level
    fn align_levels<T: Clone>(
        tree1: Box<Node<T>>,
        tree2: Box<Node<T>>,
    ) -> (Box<Node<T>>, Box<Node<T>>) {
        let max_level = tree1.level.max(tree2.level);

        let tree1 = if tree1.level < max_level {
            Self::raise_tree_level(tree1, max_level)
        } else {
            tree1
        };

        let tree2 = if tree2.level < max_level {
            Self::raise_tree_level(tree2, max_level)
        } else {
            tree2
        };

        (tree1, tree2)
    }

    /// Raise a tree to a target level by creating intermediate singleton nodes
    ///
    /// This maintains the leveling invariant (children at parent.level - 1)
    ///
    /// NOTE: The intermediate nodes created here are marked as duplicates (is_duplicate = true)
    /// because they are cloned copies of the original point, created solely for structural purposes
    /// during level alignment. This prevents k-NN queries from returning the same logical point
    /// multiple times.
    ///
    /// # Panics
    ///
    /// Panics if the tree would have to be raised by more than [`MAX_MERGE_LEVEL_GAP`]
    /// levels, because the resulting chain of placeholder nodes would be deep enough to
    /// overflow the stack in the recursive passes that follow the merge.
    fn raise_tree_level<T: Clone>(
        mut tree: Box<Node<T>>,
        target_level: i32,
    ) -> Box<Node<T>> {
        let gap = i64::from(target_level) - i64::from(tree.level);
        assert!(
            gap <= MAX_MERGE_LEVEL_GAP,
            "cannot merge cover trees whose levels differ by {} (limit {}); \
             the data spans too many orders of magnitude for this base",
            gap,
            MAX_MERGE_LEVEL_GAP
        );
        while tree.level < target_level {
            // Create new node at current level + 1 with the tree as its child
            let new_level = tree.level + 1;
            let new_point = tree.point.clone();

            // Update maxdist for the new parent
            let child_maxdist = tree.maxdist;
            let new_maxdist = child_maxdist; // Parent can reach same max distance

            tree = Box::new(Node {
                point: new_point,
                maxdist: new_maxdist,
                d_parent: 0.0,
                children: vec![tree],
                level: new_level,
                is_duplicate: true,  // Mark as duplicate - this is a cloned point for structural purposes
            });
        }

        tree
    }

    /// Create a new parent node that covers both trees
    ///
    /// Called when tree2 is too far from tree1 to be covered
    fn create_common_parent<T: Clone>(
        tree1: Box<Node<T>>,
        tree2: Box<Node<T>>,
        dist: f64,
        base: f64,
    ) -> Box<Node<T>> {
        // New parent level must be high enough to cover the distance. Both callers
        // reach here only when dist > covdist > 0.
        let new_level = crate::core::utils::level_for_distance(dist, base);

        // Raise both trees to new_level - 1
        let tree1 = Self::raise_tree_level(tree1, new_level - 1);
        let tree2 = Self::raise_tree_level(tree2, new_level - 1);

        // Use tree1's point as the new root
        let new_point = tree1.point.clone();
        let new_maxdist = dist + tree2.maxdist.max(tree1.maxdist);

        // tree1 is a self-duplicate (same point as new root), d_parent = 0.0
        // tree2 is at distance `dist` from new root
        let mut tree2 = tree2;
        tree2.d_parent = dist;

        Box::new(Node {
            point: new_point,
            maxdist: new_maxdist,
            d_parent: 0.0,
            children: vec![tree1, tree2],
            level: new_level,
            is_duplicate: true,  // This is also a cloned point for structural purposes
        })
    }

    /// Merge tree2 into tree1 (tree2 is within covering distance)
    ///
    /// This is the core merge logic following Algorithm 4
    fn merge_into<T: Clone, D: Distance<T>>(
        mut tree1: Box<Node<T>>,
        mut tree2: Box<Node<T>>,
        metric: &D,
        base: f64,
    ) -> Box<Node<T>> {
        let _guard = crate::core::utils::StackGuard::enter();
        // Step 1: Partition tree2's children into covered/not-covered by tree1
        let tree2_children = mem::take(&mut tree2.children);
        let (covered, not_covered) = Self::partition_children(&tree1, tree2_children, metric, base);

        // Step 2: Take ownership of tree1's children for modification
        let tree1_children = mem::take(&mut tree1.children);

        // Step 3: Merge covered children with tree1's children
        let (merged_children, mut leftover_covered) =
            Self::merge_children_recursively(tree1_children, covered, metric, base);

        // Step 4: Add leftover covered children and tree2's root as new children
        let mut all_children = merged_children;

        // Add leftover covered children that didn't merge with any tree1 child
        // Update d_parent since these children are moving from tree2 to tree1
        for child in leftover_covered.iter_mut() {
            child.d_parent = metric.distance(&tree1.point, &child.point);
        }
        all_children.extend(leftover_covered);

        // Insert tree2's root as a new child. If tree2's root is itself a structural
        // copy (from level alignment), its point already lives in one of its
        // descendants, so the singleton must stay marked as a duplicate.
        let d_parent = metric.distance(&tree1.point, &tree2.point);
        let tree2_singleton = Box::new(Node {
            point: tree2.point.clone(),
            maxdist: 0.0,
            d_parent,
            children: vec![],
            level: tree1.level - 1,
            is_duplicate: tree2.is_duplicate,
        });
        all_children.push(tree2_singleton);

        // Step 5: Update tree1 with new children
        tree1.children = all_children;

        // Step 6: Handle not-covered children (insert them into tree1)
        let mut result = tree1;
        for child in not_covered {
            result = Self::insert_subtree_into_tree(result, child, metric, base);
        }

        // Step 7: Recompute maxdist after all structural changes
        result.recalc_maxdist_approx(metric);

        result
    }

    /// Partition children into those covered by parent and those not covered
    ///
    /// A child is "covered" if distance(parent, child) <= parent.covdist()
    fn partition_children<T: Clone, D: Distance<T>>(
        parent: &Node<T>,
        children: Vec<Box<Node<T>>>,
        metric: &D,
        base: f64,
    ) -> (Vec<Box<Node<T>>>, Vec<Box<Node<T>>>) {
        children.into_iter().partition(|child| {
            let dist = metric.distance(&parent.point, &child.point);
            dist <= parent.covdist(base)
        })
    }

    /// Recursively merge covered children with tree1's children
    ///
    /// For each covered child, find a tree1 child within separating distance and merge them.
    /// If no matching child found, the covered child becomes a leftover.
    ///
    /// Returns (merged_children, leftover_children)
    fn merge_children_recursively<T: Clone, D: Distance<T>>(
        mut tree1_children: Vec<Box<Node<T>>>,
        covered: Vec<Box<Node<T>>>,
        metric: &D,
        base: f64,
    ) -> (Vec<Box<Node<T>>>, Vec<Box<Node<T>>>) {
        let _guard = crate::core::utils::StackGuard::enter();
        let mut leftovers = Vec::new();

        'outer: for covered_child in covered {
            // Find a tree1 child within separating distance
            for i in 0..tree1_children.len() {
                let dist = metric.distance(
                    &tree1_children[i].point,
                    &covered_child.point,
                );

                // Check if within separating distance (can be merged)
                let sepdist = tree1_children[i].sepdist(base);

                if dist <= sepdist {
                    // Merge covered_child into tree1_children[i]
                    let tree1_child = tree1_children.swap_remove(i);
                    let merged_child = Self::merge(tree1_child, covered_child, metric, base);
                    tree1_children.push(merged_child);
                    continue 'outer;
                }
            }

            // No matching tree1 child found, add to leftovers
            leftovers.push(covered_child);
        }

        (tree1_children, leftovers)
    }

    /// Insert a subtree into a tree (used for not-covered children)
    ///
    /// This flattens the subtree and inserts all points one by one
    fn insert_subtree_into_tree<T: Clone, D: Distance<T>>(
        tree: Box<Node<T>>,
        subtree: Box<Node<T>>,
        metric: &D,
        base: f64,
    ) -> Box<Node<T>> {
        let _guard = crate::core::utils::StackGuard::enter();
        // Flatten subtree to points
        let points = Self::flatten_subtree(subtree);

        // Insert each point into tree
        let mut result = tree;
        for point in points {
            result = Self::insert_point_into_tree(result, point, metric, base);
        }

        result
    }

    /// Flatten a subtree into a vector of its points.
    ///
    /// Structural duplicate nodes are skipped: their point is a copy of a point stored
    /// in one of their descendants, and re-inserting it would add a second real copy.
    fn flatten_subtree<T: Clone>(node: Box<Node<T>>) -> Vec<T> {
        let mut points = Vec::new();
        let mut stack = vec![node];
        while let Some(mut n) = stack.pop() {
            if !n.is_duplicate {
                points.push(n.point.clone());
            }
            // Reverse so children are visited in order (pre-order, as before).
            stack.extend(std::mem::take(&mut n.children).into_iter().rev());
        }
        points
    }

    /// Insert a single point into a tree (simplified insertion for merge)
    ///
    /// This is a basic insertion that maintains invariants.
    fn insert_point_into_tree<T: Clone, D: Distance<T>>(
        mut tree: Box<Node<T>>,
        point: T,
        metric: &D,
        base: f64,
    ) -> Box<Node<T>> {
        let _guard = crate::core::utils::StackGuard::enter();
        let dist = metric.distance(&tree.point, &point);
        let tree_level = tree.level;

        // If point is too far, raise tree level
        if dist > tree.covdist(base) {
            return Self::create_common_parent(
                tree,
                Box::new(Node::new(point, tree_level, false)),
                dist,
                base,
            );
        }

        // Exact duplicates of this node's point go into a balanced group of copies
        // below it (see `DUPLICATE_FANOUT`), so long runs of copies cannot build a
        // deep chain.
        if dist == 0.0 {
            if let Some(i) = crate::core::utils::place_duplicate(&tree, &point, metric) {
                // Rotate the chosen copy to the back so ties go round-robin.
                let child = tree.children.remove(i);
                let child = Self::insert_point_into_tree(child, point, metric, base);
                tree.children.push(child);
                return tree;
            }
            let level = tree.level - 1;
            tree.children.push(Box::new(Node::new(point, level, false)));
            return tree;
        }

        // Try to find a child to recurse into
        for i in 0..tree.children.len() {
            let child_dist = metric.distance(&tree.children[i].point, &point);

            if child_dist <= tree.children[i].covdist(base) {
                // Recurse into this child
                let child = tree.children.swap_remove(i);
                let new_child = Self::insert_point_into_tree(child, point, metric, base);
                tree.children.push(new_child);
                tree.maxdist = tree.maxdist.max(dist);
                return tree;
            }
        }

        // No suitable child, add as new child
        let mut new_child = Box::new(Node::new(point, tree.level - 1, false));
        new_child.d_parent = dist;
        tree.children.push(new_child);
        tree.maxdist = tree.maxdist.max(dist);

        tree
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq)]
    struct Point(f64);

    struct SimpleDistance;
    impl Distance<Point> for SimpleDistance {
        fn distance(&self, p: &Point, q: &Point) -> f64 {
            (p.0 - q.0).abs()
        }
    }

    #[test]
    fn test_align_levels_equal() {
        let tree1 = Box::new(Node::new(Point(0.0), 5, false));
        let tree2 = Box::new(Node::new(Point(1.0), 5, false));

        let (t1, t2) = MergeImpl::align_levels(tree1, tree2);
        assert_eq!(t1.level, 5);
        assert_eq!(t2.level, 5);
    }

    #[test]
    fn test_align_levels_different() {
        let tree1 = Box::new(Node::new(Point(0.0), 3, false));
        let tree2 = Box::new(Node::new(Point(1.0), 5, false));

        let (t1, t2) = MergeImpl::align_levels(tree1, tree2);
        assert_eq!(t1.level, 5);
        assert_eq!(t2.level, 5);
    }

    #[test]
    fn test_raise_tree_level() {
        let tree = Box::new(Node::new(Point(0.0), 2, false));
        let raised = MergeImpl::raise_tree_level(tree, 5);

        assert_eq!(raised.level, 5);
        assert_eq!(raised.point, Point(0.0));
    }

    #[test]
    fn test_partition_children() {
        let parent = Node::new(Point(0.0), 5, false);
        let metric = SimpleDistance;

        let child1 = Box::new(Node::new(Point(1.0), 4, false)); // Within covdist
        let child2 = Box::new(Node::new(Point(10.0), 4, false)); // Outside covdist

        let children = vec![child1, child2];
        let (covered, not_covered) = MergeImpl::partition_children(&parent, children, &metric, 1.3);

        assert_eq!(covered.len(), 1);
        assert_eq!(not_covered.len(), 1);
    }
}
