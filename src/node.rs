//! Node Structure Module
//!
//! This module defines the `Node` struct, which is the fundamental building block of the cover tree.
//! Each node represents a single data point and maintains structural information needed for the
//! three cover tree invariants.
//!
//! # What is a Node?
//!
//! In a cover tree, each node contains:
//! - A data point (the actual point in the metric space)
//! - A level (integer determining covering and separating distances)
//! - Children (other nodes at the next level down)
//! - maxdist (maximum distance to any descendant - cached for query pruning)
//!
//! # The Three Invariants
//!
//! Cover tree nodes must maintain three invariants:
//!
//! 1. **Leveling Invariant**: Each child is exactly one level below its parent
//!    - If parent has level(p), then child has level(p) - 1
//!
//! 2. **Covering Invariant**: Children are within covering distance of parent
//!    - For each child c of parent p: d(p, c) ≤ covdist(p) = base^level(p)
//!
//! 3. **Separating Invariant**: Children are separated from each other
//!    - For any two distinct children c1, c2 of parent p: d(c1, c2) > sepdist(p) = base^(level(p)-1)
//!
//! # Configurable Base Value
//!
//! The base value (typically 1.3 from the paper, but configurable) determines the growth
//! rate of covering and separating distances. Using base 1.3 instead of 2.0 empirically
//! reduces distance computations by 10-71% on benchmark datasets. This value is stored
//! per-tree (not per-node) and passed to `covdist()` / `sepdist()` as needed.

use crate::distance::Distance;

/// Tree structure quality metrics for diagnosing pruning effectiveness.
///
/// Key metric: `avg_maxdist_covdist_ratio` — measures how tight pruning bounds
/// are relative to theoretical maximum. Lower values = tighter pruning.
/// Trees built with `from_batch()` typically have higher ratios than
/// incrementally-built trees.
#[derive(Debug, Clone)]
pub struct TreeStats {
    pub total_nodes: usize,
    pub leaf_count: usize,
    pub max_depth: usize,
    pub avg_children: f64,
    pub max_children: usize,
    pub avg_maxdist: f64,
    pub max_maxdist: f64,
    pub children_histogram: Vec<usize>,
    /// Average of maxdist/covdist for internal nodes. Lower = tighter pruning.
    pub avg_maxdist_covdist_ratio: f64,
    pub level_span: i32,
}

/// A node in the cover tree.
///
/// Each node represents a single point in the metric space and maintains the structural
/// information needed for cover tree operations.
///
/// # Type Parameters
///
/// * `T` - The type of point stored. Must implement `Clone` so we can duplicate points
///   when needed (e.g., during tree restructuring).
///
/// # Fields (ordered for cache locality)
///
/// Hot fields (accessed together during queries) are first:
/// * `point` - The data point this node represents. **Ownership**: The node owns this point.
/// * `maxdist` - Maximum distance from this point to any descendant (cached for efficiency).
/// * `d_parent` - Distance from parent's point to this node's point (cached for dual-tree).
///
/// Warm fields (accessed during tree traversal):
/// * `children` - Child nodes at level-1. **Ownership**: The node owns its children via Box.
///
/// Cold fields (accessed infrequently):
/// * `level` - The integer level of this node. Determines covdist and sepdist.
/// * `is_duplicate` - Internal flag marking algorithm-created duplicates (used for k-NN deduplication).
///
/// # Internal Implementation Detail
///
/// The `is_duplicate` field marks algorithm-created duplicate nodes that arise during tree
/// merging and level alignment. These duplicates are automatically skipped during k-NN queries
/// to avoid returning the same logical point multiple times. This field is set internally by
/// tree operations and should not be modified by users.
///
/// # Examples
///
/// ```
/// use rustknn::Node;
///
/// // Create a new node at level 3
/// let node = Node::new(42, 3, false);
/// assert_eq!(node.level, 3);
/// assert_eq!(node.point, 42);
/// assert_eq!(node.children.len(), 0);
///
/// // Calculate covering distance (base is passed as parameter)
/// let covdist = node.covdist(1.3);
/// assert!((covdist - 1.3_f64.powi(3)).abs() < 1e-10);
/// ```
pub struct Node<T: Clone> {
    // --- HOT fields: accessed together during queries ---

    /// The data point this node represents.
    ///
    /// **Ownership**: The node owns this point. It's moved into the node when created.
    pub point: T,

    /// Maximum distance from this node's point to any of its descendants.
    ///
    /// This value is cached for efficiency - it's used in the pruning condition during
    /// nearest neighbor queries (Algorithm 1, line 4).
    ///
    /// **Invariant**: maxdist <= 2 * base^(level + 1)
    /// **Updates**: Updated incrementally when children are added, or recalculated when needed.
    pub maxdist: f64,

    /// Distance from parent's point to this node's point: d(parent.point, self.point).
    ///
    /// Cached during insertion to avoid recomputation during dual-tree traversal.
    /// For the root node (no parent), this is 0.0.
    ///
    /// Used in Phase 1 of the both-internal case to provide d(qp, qc_i) distances
    /// for triangle-inequality pre-pruning, eliminating `nq` distance computations
    /// per both-internal node pair visited.
    pub d_parent: f64,

    // --- WARM fields: accessed during tree traversal ---

    /// The children of this node.
    ///
    /// **Ownership**: Each child is heap-allocated (Box) and owned by this Vec.
    /// **Why Box?**: Prevents infinite size problem with recursive types.
    /// **Invariant**: All children must have level = parent.level - 1
    pub children: Vec<Box<Node<T>>>,

    // --- COLD fields: accessed infrequently ---

    /// The integer level of this node in the tree.
    ///
    /// Higher levels are near the root, lower levels are near the leaves.
    /// The root can have any level (determined by data), and children are at parent.level - 1.
    pub level: i32,

    /// Internal flag marking algorithm-created duplicate nodes.
    ///
    /// During tree merging (parallel construction) and level alignment, points are cloned
    /// to create intermediate nodes. These clones appear as separate entities but represent
    /// the same logical point. This flag distinguishes them:
    ///
    /// - `false`: Original user-inserted point (default, always returned in k-NN results)
    /// - `true`: Algorithm-created duplicate (skipped in k-NN results to avoid duplicates)
    ///
    /// **Important**: User's intentional duplicate values (two points with the same value)
    /// both have `is_duplicate = false` and both are correctly returned in k-NN results.
    ///
    /// This field is set internally by tree operations and should not be modified by users.
    pub is_duplicate: bool,
}

impl<T: Clone> Drop for Node<T> {
    /// Frees the subtree iteratively. The compiler-generated drop would recurse once per
    /// level and could overflow the stack on a very deep tree.
    fn drop(&mut self) {
        let mut stack = std::mem::take(&mut self.children);
        while let Some(mut child) = stack.pop() {
            stack.append(&mut child.children);
        }
    }
}

impl<T: Clone + std::fmt::Debug> std::fmt::Debug for Node<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let _guard = crate::core::utils::StackGuard::enter();
        f.debug_struct("Node")
            .field("point", &self.point)
            .field("maxdist", &self.maxdist)
            .field("d_parent", &self.d_parent)
            .field("children", &self.children)
            .field("level", &self.level)
            .field("is_duplicate", &self.is_duplicate)
            .finish()
    }
}

impl<T: Clone> Node<T> {
    /// Creates a new node with the given point and level.
    ///
    /// The node starts with no children and maxdist = 0.0.
    ///
    /// # Arguments
    ///
    /// * `point` - The data point for this node. **Ownership**: This function takes ownership
    ///   of the point (it's moved in). The point now belongs to the node.
    /// * `level` - The integer level for this node. Determines covdist and sepdist.
    /// * `is_duplicate` - Whether this node is an algorithm-created duplicate (default: false).
    ///   Set to `true` only when cloning points during tree merging/level alignment.
    ///
    /// # Returns
    ///
    /// A new `Node<T>` with the given point and level, no children, and maxdist = 0.0.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustknn::Node;
    ///
    /// // Create a node with an integer point
    /// let node = Node::new(42, 5, false);
    /// assert_eq!(node.point, 42);
    /// assert_eq!(node.level, 5);
    /// assert_eq!(node.children.len(), 0);
    /// assert_eq!(node.maxdist, 0.0);
    /// assert_eq!(node.is_duplicate, false);
    /// ```
    ///
    /// ```
    /// use rustknn::Node;
    ///
    /// #[derive(Clone)]
    /// struct Point2D { x: f64, y: f64 }
    ///
    /// let point = Point2D { x: 1.0, y: 2.0 };
    /// let node = Node::new(point, 0, false);
    /// // Can't use `point` here anymore - it was moved into the node!
    /// // println!("{}", point.x); // This would be a compile error
    ///
    /// // But we can access it through the node:
    /// println!("{}", node.point.x); // This works!
    /// ```
    pub fn new(point: T, level: i32, is_duplicate: bool) -> Self {
        Node {
            point,      // Shorthand for `point: point` (field name matches parameter name)
            maxdist: 0.0,
            d_parent: 0.0,
            children: Vec::new(),
            level,      // Shorthand for `level: level`
            is_duplicate,
        }
    }

    /// Computes the covering distance for this node.
    ///
    /// The covering distance is covdist(p) = base^level(p). This determines how far children
    /// can be from their parent. The covering invariant requires that all children are within
    /// this distance.
    ///
    /// # Base Value
    ///
    /// The base value is passed as a parameter (stored per-tree, not per-node). The paper used
    /// base 2 originally, but found that base 1.3 works better in practice (see paper footnote
    /// page 2), reducing distance computations by 10-71% on benchmark datasets.
    ///
    /// # Arguments
    ///
    /// * `base` - The base value for distance calculations (typically 1.3).
    ///
    /// # Returns
    ///
    /// The covering distance as a 64-bit float: base^(self.level)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustknn::Node;
    ///
    /// let node = Node::new(0, 5, false);
    /// let covdist = node.covdist(1.3);
    /// let expected = 1.3_f64.powi(5); // 1.3^5 ≈ 3.71293
    /// assert!((covdist - expected).abs() < 1e-10);
    /// ```
    ///
    /// ```
    /// use rustknn::Node;
    ///
    /// let node = Node::new(0, 0, false);
    /// assert_eq!(node.covdist(1.3), 1.0); // 1.3^0 = 1.0
    ///
    /// let node = Node::new(0, -2, false);
    /// let expected = 1.3_f64.powi(-2); // 1.3^(-2) ≈ 0.5917
    /// assert!((node.covdist(1.3) - expected).abs() < 1e-10);
    ///
    /// // Try a different base value
    /// let node = Node::new(0, 3, false);
    /// assert_eq!(node.covdist(2.0), 8.0); // 2.0^3 = 8.0
    /// ```
    pub fn covdist(&self, base: f64) -> f64 {
        base.powi(self.level)
    }

    /// Computes the separation distance for this node.
    ///
    /// The separation distance is sepdist(p) = base^(level(p) - 1). This determines how far apart
    /// children must be from each other. The separating invariant requires that all pairs of
    /// distinct children are MORE than this distance apart.
    ///
    /// # Arguments
    ///
    /// * `base` - The base value for distance calculations (typically 1.3).
    ///
    /// # Returns
    ///
    /// The separation distance as a 64-bit float: base^(self.level - 1)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustknn::Node;
    ///
    /// let node = Node::new(0, 5, false);
    /// let sepdist = node.sepdist(1.3);
    /// let expected = 1.3_f64.powi(4); // 1.3^(5-1) = 1.3^4
    /// assert!((sepdist - expected).abs() < 1e-10);
    /// ```
    ///
    /// Note: sepdist(p) = covdist(p) / base (one level lower)
    pub fn sepdist(&self, base: f64) -> f64 {
        base.powi(self.level - 1)
    }

    /// Updates maxdist with an upper bound based on a newly added child.
    ///
    /// When adding a new child, we need to update the parent's maxdist value. The maxdist
    /// of a node is **defined** as the exact maximum distance from that node to any of its
    /// descendants: `maxdist(p) = max{d(p,q) : q ∈ descendants(p)}`.
    ///
    /// However, computing this exactly during insertion would be expensive (O(subtree size) per insertion).
    /// Instead, this method maintains an **upper bound** using the triangle inequality.
    ///
    /// # Formula (Upper Bound via Triangle Inequality)
    ///
    /// `maxdist(parent) ≤ max over all children c of: d(parent, c) + maxdist(c)`
    ///
    /// This uses the triangle inequality: `d(parent, descendant) ≤ d(parent, child) + d(child, descendant)`.
    /// The actual maximum might be smaller if descendants are not collinear with their ancestors.
    ///
    /// # Important: This Produces an Upper Bound
    ///
    /// This method produces an **upper bound**, not the exact maximum. The bound can
    /// overestimate the true maxdist, especially in deep trees where the approximation accumulates.
    ///
    /// **Upper bounds are safe**: Query pruning with overestimated maxdist is conservative (may
    /// traverse more nodes than necessary) but never incorrect. Insertion logic that uses maxdist
    /// for pruning (e.g., Nearest Ancestor rebalancing) remains correct with upper bounds.
    ///
    /// **For optimal query performance**, call `CoverTree::recompute_maxdist()` after batch
    /// insertions to compute exact maxdist values. Exact bounds enable better pruning (10-30% fewer
    /// distance computations during queries, per the paper). This matches HLearn's `setMaxDescendentDistance`.
    ///
    /// # Arguments
    ///
    /// * `child_dist` - The distance from this node's point to the child's point
    /// * `child_maxdist` - The maxdist value of the child being added (may also be an upper bound)
    ///
    /// # Examples
    ///
    /// ```
    /// use rustknn::Node;
    ///
    /// let mut parent = Node::new(0, 0, false);
    /// assert_eq!(parent.maxdist, 0.0);
    ///
    /// // Add a child at distance 5 with its own maxdist of 3
    /// parent.update_maxdist_approx(5.0, 3.0);
    /// assert_eq!(parent.maxdist, 8.0); // Upper bound: 5 + 3
    ///
    /// // Add another child at distance 2 with maxdist of 1
    /// parent.update_maxdist_approx(2.0, 1.0);
    /// assert_eq!(parent.maxdist, 8.0); // Still 8 (doesn't decrease)
    ///
    /// // Add a farther child
    /// parent.update_maxdist_approx(6.0, 4.0);
    /// assert_eq!(parent.maxdist, 10.0); // Upper bound: 6 + 4
    ///
    /// // Note: The actual maximum distance might be less than 10.0
    /// // Call tree.recompute_maxdist() to get exact values
    /// ```
    pub fn update_maxdist_approx(&mut self, child_dist: f64, child_maxdist: f64) {
        let total_dist = child_dist + child_maxdist;
        if total_dist > self.maxdist {
            self.maxdist = total_dist;
        }
    }

    /// Recalculates maxdist from scratch by examining direct children only.
    ///
    /// Normally, maxdist is updated incrementally as children are added. However, if we remove
    /// a child (especially the one that had the maximum distance), we need to recalculate
    /// maxdist from scratch.
    ///
    /// # Important: This Also Uses Triangle Inequality (Upper Bound)
    ///
    /// This method iterates through all **direct children**, computes the distance to each, adds that
    /// child's maxdist, and takes the maximum: `maxdist(parent) = max(dist(parent, c) + maxdist(c))`.
    ///
    /// Like `update_maxdist_approx()`, this uses the **triangle inequality** and produces an **upper bound**,
    /// not the exact maximum. It's more accurate than incremental updates (since it checks all children),
    /// but still uses each child's maxdist (which may itself be an upper bound).
    ///
    /// **For exact maxdist computation** that examines ALL descendants (not just direct children),
    /// use `CoverTree::recompute_maxdist()` which recursively computes exact values for the
    /// entire tree. This is more expensive but provides tight bounds for optimal query performance.
    ///
    /// # Arguments
    ///
    /// * `metric` - The distance metric to use. **Borrowed**: We borrow the metric, don't consume it.
    ///
    /// # Generic Method
    ///
    /// This method has its own generic parameter `D` separate from the struct's `T` parameter.
    /// `D` must implement the `Distance<T>` trait.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustknn::{Node, Distance};
    ///
    /// struct EuclideanDistance;
    /// impl Distance<f64> for EuclideanDistance {
    ///     fn distance(&self, p: &f64, q: &f64) -> f64 {
    ///         (p - q).abs()
    ///     }
    /// }
    ///
    /// let mut parent = Node::new(0.0, 1, false);
    /// parent.children.push(Box::new(Node::new(5.0, 0, false)));
    /// parent.children.push(Box::new(Node::new(3.0, 0, false)));
    ///
    /// let metric = EuclideanDistance;
    /// parent.recalc_maxdist_approx(&metric);
    ///
    /// // maxdist should be 5.0 (distance to farthest child)
    /// assert_eq!(parent.maxdist, 5.0);
    /// ```
    pub fn recalc_maxdist_approx<D: Distance<T>>(&mut self, metric: &D) {
        self.maxdist = 0.0;
        for child in &self.children {
            let dist = metric.distance(&self.point, &child.point);
            let total = dist + child.maxdist;
            if total > self.maxdist {
                self.maxdist = total;
            }
        }
    }

    /// Collect tree quality statistics by walking the subtree rooted at this node.
    ///
    /// Returns a `TreeStats` summarizing structure quality metrics. The key metric
    /// is `avg_maxdist_covdist_ratio`: lower values indicate tighter pruning bounds.
    pub fn collect_stats(&self, base: f64) -> TreeStats {
        let mut total_nodes: usize = 0;
        let mut leaf_count: usize = 0;
        let mut max_depth: usize = 0;
        let mut max_children: usize = 0;
        let mut total_children: usize = 0;  // sum of children counts (for internal nodes)
        let mut internal_count: usize = 0;
        let mut sum_maxdist: f64 = 0.0;
        let mut max_maxdist: f64 = 0.0;
        let mut sum_ratio: f64 = 0.0;
        let mut ratio_count: usize = 0;
        let mut min_level: i32 = self.level;
        let mut max_level: i32 = self.level;
        // Histogram: children_histogram[i] = count of nodes with i children
        let mut children_histogram: Vec<usize> = Vec::new();

        // Iterative DFS to avoid stack overflow on deep trees
        let mut stack: Vec<(&Node<T>, usize)> = vec![(self, 0)];

        while let Some((node, depth)) = stack.pop() {
            total_nodes += 1;
            if depth > max_depth {
                max_depth = depth;
            }
            if node.level < min_level {
                min_level = node.level;
            }
            if node.level > max_level {
                max_level = node.level;
            }

            let nc = node.children.len();
            if nc == 0 {
                leaf_count += 1;
            } else {
                internal_count += 1;
                total_children += nc;
                if nc > max_children {
                    max_children = nc;
                }
            }

            // Grow histogram if needed
            if nc >= children_histogram.len() {
                children_histogram.resize(nc + 1, 0);
            }
            children_histogram[nc] += 1;

            sum_maxdist += node.maxdist;
            if node.maxdist > max_maxdist {
                max_maxdist = node.maxdist;
            }

            // maxdist/covdist ratio for internal nodes
            if nc > 0 {
                let covdist = node.covdist(base);
                if covdist > 0.0 {
                    sum_ratio += node.maxdist / covdist;
                    ratio_count += 1;
                }
            }

            for child in &node.children {
                stack.push((child, depth + 1));
            }
        }

        TreeStats {
            total_nodes,
            leaf_count,
            max_depth,
            avg_children: if internal_count > 0 {
                total_children as f64 / internal_count as f64
            } else {
                0.0
            },
            max_children,
            avg_maxdist: if total_nodes > 0 {
                sum_maxdist / total_nodes as f64
            } else {
                0.0
            },
            max_maxdist,
            children_histogram,
            avg_maxdist_covdist_ratio: if ratio_count > 0 {
                sum_ratio / ratio_count as f64
            } else {
                0.0
            },
            level_span: max_level - min_level,
        }
    }
}

// Unit tests for the Node struct
#[cfg(test)]
mod tests {
    use super::*;

    // Simple distance metric for testing
    struct SimpleDistance;
    impl Distance<f64> for SimpleDistance {
        fn distance(&self, p: &f64, q: &f64) -> f64 {
            (p - q).abs()
        }
    }

    #[test]
    fn test_node_creation() {
        let node = Node::new(42, 3, false);
        assert_eq!(node.point, 42);
        assert_eq!(node.level, 3);
        assert_eq!(node.children.len(), 0);
        assert_eq!(node.maxdist, 0.0);
        assert_eq!(node.is_duplicate, false);
    }

    #[test]
    fn test_node_creation_different_base() {
        let node = Node::new(10, 2, false);
        assert_eq!(node.point, 10);
        assert_eq!(node.level, 2);
        // covdist should be 2.0^2 = 4.0
        assert_eq!(node.covdist(2.0), 4.0);
    }

    #[test]
    fn test_covdist_calculation() {
        let node = Node::new(0, 0, false);
        assert_eq!(node.covdist(1.3), 1.0); // 1.3^0 = 1

        let node = Node::new(0, 5, false);
        let expected = 1.3_f64.powi(5);
        assert!((node.covdist(1.3) - expected).abs() < 1e-10);

        let node = Node::new(0, -2, false);
        let expected = 1.3_f64.powi(-2);
        assert!((node.covdist(1.3) - expected).abs() < 1e-10);

        // Test with base 2.0
        let node = Node::new(0, 3, false);
        assert_eq!(node.covdist(2.0), 8.0); // 2^3 = 8
    }

    #[test]
    fn test_sepdist_calculation() {
        let node = Node::new(0, 5, false);
        let expected = 1.3_f64.powi(4); // 1.3^(5-1)
        assert!((node.sepdist(1.3) - expected).abs() < 1e-10);

        // Verify relationship: sepdist = covdist / base
        let base = 1.3;
        let covdist = node.covdist(base);
        let sepdist = node.sepdist(base);
        assert!((sepdist * base - covdist).abs() < 1e-10);

        // Test with base 2.0
        let node = Node::new(0, 4, false);
        assert_eq!(node.sepdist(2.0), 8.0); // 2^(4-1) = 2^3 = 8
    }

    #[test]
    fn test_maxdist_update() {
        let mut node = Node::new(0, 0, false);
        assert_eq!(node.maxdist, 0.0);

        // First update
        node.update_maxdist_approx(2.0, 3.0);
        assert_eq!(node.maxdist, 5.0);

        // Update with smaller value (shouldn't change)
        node.update_maxdist_approx(1.0, 2.0);
        assert_eq!(node.maxdist, 5.0);

        // Update with larger value
        node.update_maxdist_approx(3.0, 4.0);
        assert_eq!(node.maxdist, 7.0);
    }

    #[test]
    fn test_maxdist_recalculation() {
        let mut parent = Node::new(0.0, 1, false);

        // Add some children
        let mut child1 = Node::new(5.0, 0, false);
        child1.maxdist = 2.0;
        parent.children.push(Box::new(child1));

        let mut child2 = Node::new(3.0, 0, false);
        child2.maxdist = 1.0;
        parent.children.push(Box::new(child2));

        let metric = SimpleDistance;
        parent.recalc_maxdist_approx(&metric);

        // Should be max(5+2, 3+1) = max(7, 4) = 7
        assert_eq!(parent.maxdist, 7.0);
    }

    #[test]
    fn test_maxdist_with_no_children() {
        let mut node = Node::new(0.0, 0, false);
        node.maxdist = 5.0; // Set to some value

        let metric = SimpleDistance;
        node.recalc_maxdist_approx(&metric);

        // Should reset to 0 (no children)
        assert_eq!(node.maxdist, 0.0);
    }
}
