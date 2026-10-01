//! Unified Cover Tree Interface
//!
//! This module provides a unified `CoverTree` enum that wraps either the Simplified
//! or Nearest Ancestor variant, allowing users to choose their preferred variant
//! through a single API.
//!
//! # Design
//!
//! `CoverTree` is a thin wrapper (zero-cost abstraction) that delegates all operations
//! to the underlying implementation. The actual implementations live in:
//! - `SimplifiedCoverTree` in `src/simplified/tree.rs`
//! - `NACoverTree` in `src/nearest_ancestor/tree.rs`
//!
//! # Usage
//!
//! ```rust
//! use rustknn::{CoverTree, Distance};
//!
//! // Define your point type
//! #[derive(Clone)]
//! struct Point2D { x: f64, y: f64 }
//!
//! // Implement the Distance trait
//! struct EuclideanDistance;
//! impl Distance<Point2D> for EuclideanDistance {
//!     fn distance(&self, p: &Point2D, q: &Point2D) -> f64 {
//!         let dx = p.x - q.x;
//!         let dy = p.y - q.y;
//!         (dx * dx + dy * dy).sqrt()
//!     }
//! }
//!
//! // Create a Simplified Cover Tree (default)
//! let metric = EuclideanDistance;
//! let mut tree = CoverTree::new(metric, 1.3);
//! tree.insert(Point2D { x: 0.0, y: 0.0 });
//! tree.insert(Point2D { x: 1.0, y: 1.0 });
//!
//! // Query for nearest neighbor
//! let query = Point2D { x: 0.5, y: 0.5 };
//! let nearest = tree.find_nearest(&query);
//! ```
//!
//! Or create a Nearest Ancestor variant:
//!
//! ```rust
//! use rustknn::{CoverTree, Distance};
//!
//! struct SimpleDistance;
//! impl Distance<f64> for SimpleDistance {
//!     fn distance(&self, p: &f64, q: &f64) -> f64 {
//!         (p - q).abs()
//!     }
//! }
//!
//! // Create a Nearest Ancestor Cover Tree (better query performance)
//! let mut tree = CoverTree::new_nearest_ancestor(SimpleDistance, 1.3);
//! tree.insert(0.0);
//! tree.insert(100.0);
//! tree.insert(50.0);
//!
//! let nearest = tree.find_nearest(&60.0);
//! ```
//!
//! # Alternative: Direct API
//!
//! You can also use the specific implementations directly for better type clarity:
//!
//! ```rust,ignore
//! use rustknn::simplified::SimplifiedCoverTree;
//! use rustknn::nearest_ancestor::NACoverTree;
//!
//! let mut simple_tree = SimplifiedCoverTree::new(metric, 1.3);
//! let mut na_tree = NACoverTree::new(metric, 1.3);
//! ```

use crate::distance::Distance;
use crate::node::Node;
use crate::simplified::SimplifiedCoverTree;
use crate::nearest_ancestor::NACoverTree;

/// Specifies which variant of the cover tree to use.
///
/// # Variants
///
/// * `Simplified` - Exactly n nodes, no rebalancing. Faster construction.
/// * `NearestAncestor` - Adds rebalancing to maintain nearest ancestor invariant.
///   Slower construction but 10-30% fewer distance computations during queries.
///
/// # Choosing a Variant
///
/// **Use Simplified when:**
/// - Construction speed is critical
/// - Dataset is small to medium (< 100K points)
/// - Query performance is "good enough"
///
/// **Use NearestAncestor when:**
/// - Query performance is critical (queries dominate workload)
/// - Large datasets (> 100K points)
/// - Willing to pay construction cost for query speedup
///
/// **Benchmark both variants on your specific dataset to decide!**
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreeVariant {
    /// Simplified cover tree: n nodes for n points, no rebalancing
    Simplified,
    /// Nearest ancestor cover tree: maintains nearest ancestor invariant via rebalancing
    NearestAncestor,
}

/// A cover tree for efficient nearest neighbor search.
///
/// This is a unified wrapper that can hold either a `SimplifiedCoverTree` or
/// `NACoverTree`. All operations are delegated to the underlying implementation.
///
/// # Type Parameters
///
/// * `T` - The type of points stored. Must implement `Clone`.
/// * `D` - The distance metric. Must implement `Distance<T>`.
///
/// # Examples
///
/// ```rust
/// use rustknn::{CoverTree, Distance};
///
/// struct SimpleDistance;
/// impl Distance<f64> for SimpleDistance {
///     fn distance(&self, p: &f64, q: &f64) -> f64 {
///         (p - q).abs()
///     }
/// }
///
/// // Create Simplified variant (default)
/// let mut tree = CoverTree::new(SimpleDistance, 1.3);
/// tree.insert(10.0);
/// tree.insert(20.0);
/// assert_eq!(tree.len(), 2);
///
/// // Query
/// let nearest = tree.find_nearest(&15.0);
/// assert!(nearest.is_some());
/// ```
///
/// For Nearest Ancestor variant:
///
/// ```rust
/// use rustknn::{CoverTree, Distance};
///
/// struct SimpleDistance;
/// impl Distance<f64> for SimpleDistance {
///     fn distance(&self, p: &f64, q: &f64) -> f64 {
///         (p - q).abs()
///     }
/// }
///
/// // Create Nearest Ancestor variant
/// let mut tree = CoverTree::new_nearest_ancestor(SimpleDistance, 1.3);
/// tree.insert(10.0);
/// tree.insert(20.0);
/// ```
#[derive(Debug)]
pub enum CoverTree<T: Clone, D: Distance<T>> {
    /// Simplified variant
    Simplified(SimplifiedCoverTree<T, D>),
    /// Nearest Ancestor variant
    NearestAncestor(NACoverTree<T, D>),
}

impl<T: Clone, D: Distance<T>> CoverTree<T, D> {
    /// Creates a new empty Simplified cover tree.
    ///
    /// This is the default constructor and creates a Simplified variant,
    /// which has faster construction but may perform more distance
    /// computations during queries.
    ///
    /// # Arguments
    ///
    /// * `metric` - The distance function to use
    /// * `base` - The base value for distance calculations (typically 1.3)
    ///
    /// # Returns
    ///
    /// A new empty `CoverTree` with Simplified variant.
    ///
    /// # Panics
    ///
    /// Panics if `base` is not finite or is below [`MIN_BASE`](crate::MIN_BASE).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustknn::{CoverTree, Distance};
    ///
    /// struct SimpleDistance;
    /// impl Distance<f64> for SimpleDistance {
    ///     fn distance(&self, p: &f64, q: &f64) -> f64 {
    ///         (p - q).abs()
    ///     }
    /// }
    ///
    /// let tree = CoverTree::new(SimpleDistance, 1.3);
    /// assert_eq!(tree.len(), 0);
    /// ```
    pub fn new(metric: D, base: f64) -> Self {
        CoverTree::Simplified(SimplifiedCoverTree::new(metric, base))
    }

    /// Creates a new empty Nearest Ancestor cover tree.
    ///
    /// This creates a Nearest Ancestor variant, which has slower construction
    /// (due to rebalancing) but performs 10-30% fewer distance computations
    /// during queries compared to the Simplified variant.
    ///
    /// # Arguments
    ///
    /// * `metric` - The distance function to use
    /// * `base` - The base value for distance calculations (typically 1.3)
    ///
    /// # Returns
    ///
    /// A new empty `CoverTree` with Nearest Ancestor variant.
    ///
    /// # Panics
    ///
    /// Panics if `base` is not finite or is below [`MIN_BASE`](crate::MIN_BASE).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustknn::{CoverTree, Distance};
    ///
    /// struct SimpleDistance;
    /// impl Distance<f64> for SimpleDistance {
    ///     fn distance(&self, p: &f64, q: &f64) -> f64 {
    ///         (p - q).abs()
    ///     }
    /// }
    ///
    /// let tree = CoverTree::new_nearest_ancestor(SimpleDistance, 1.3);
    /// assert_eq!(tree.len(), 0);
    /// ```
    pub fn new_nearest_ancestor(metric: D, base: f64) -> Self {
        CoverTree::NearestAncestor(NACoverTree::new(metric, base))
    }

    /// Creates a new cover tree with the specified variant.
    ///
    /// # Arguments
    ///
    /// * `metric` - The distance function to use
    /// * `base` - The base value for distance calculations
    /// * `variant` - The tree variant to use (Simplified or NearestAncestor)
    ///
    /// # Panics
    ///
    /// Panics if `base` is not finite or is below [`MIN_BASE`](crate::MIN_BASE).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustknn::{CoverTree, TreeVariant, Distance};
    ///
    /// struct SimpleDistance;
    /// impl Distance<f64> for SimpleDistance {
    ///     fn distance(&self, p: &f64, q: &f64) -> f64 {
    ///         (p - q).abs()
    ///     }
    /// }
    ///
    /// // Create with custom options
    /// let tree = CoverTree::with_options(
    ///     SimpleDistance,
    ///     1.3,
    ///     TreeVariant::Simplified,
    /// );
    /// ```
    pub fn with_options(metric: D, base: f64, variant: TreeVariant) -> Self {
        match variant {
            TreeVariant::Simplified => {
                CoverTree::Simplified(SimplifiedCoverTree::new(metric, base))
            }
            TreeVariant::NearestAncestor => {
                CoverTree::NearestAncestor(NACoverTree::new(metric, base))
            }
        }
    }

    /// Returns the number of points in the tree.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustknn::{CoverTree, Distance};
    ///
    /// struct SimpleDistance;
    /// impl Distance<f64> for SimpleDistance {
    ///     fn distance(&self, p: &f64, q: &f64) -> f64 {
    ///         (p - q).abs()
    ///     }
    /// }
    ///
    /// let mut tree = CoverTree::new(SimpleDistance, 1.3);
    /// assert_eq!(tree.len(), 0);
    ///
    /// tree.insert(10.0);
    /// assert_eq!(tree.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        match self {
            CoverTree::Simplified(tree) => tree.len(),
            CoverTree::NearestAncestor(tree) => tree.len(),
        }
    }

    /// Returns true if the tree contains no points.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustknn::{CoverTree, Distance};
    ///
    /// struct SimpleDistance;
    /// impl Distance<f64> for SimpleDistance {
    ///     fn distance(&self, p: &f64, q: &f64) -> f64 {
    ///         (p - q).abs()
    ///     }
    /// }
    ///
    /// let mut tree = CoverTree::new(SimpleDistance, 1.3);
    /// assert!(tree.is_empty());
    ///
    /// tree.insert(10.0);
    /// assert!(!tree.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        match self {
            CoverTree::Simplified(tree) => tree.is_empty(),
            CoverTree::NearestAncestor(tree) => tree.is_empty(),
        }
    }

    /// Returns a reference to the root node, if it exists.
    ///
    /// This is primarily for testing and validation purposes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustknn::{CoverTree, Distance};
    ///
    /// struct SimpleDistance;
    /// impl Distance<f64> for SimpleDistance {
    ///     fn distance(&self, p: &f64, q: &f64) -> f64 {
    ///         (p - q).abs()
    ///     }
    /// }
    ///
    /// let mut tree = CoverTree::new(SimpleDistance, 1.3);
    /// assert!(tree.root_node().is_none());
    ///
    /// tree.insert(10.0);
    /// assert!(tree.root_node().is_some());
    /// ```
    pub fn root_node(&self) -> Option<&Node<T>> {
        match self {
            CoverTree::Simplified(tree) => tree.root_node(),
            CoverTree::NearestAncestor(tree) => tree.root_node(),
        }
    }

    /// Returns a reference to the distance metric.
    pub fn metric(&self) -> &D {
        match self {
            CoverTree::Simplified(tree) => tree.metric(),
            CoverTree::NearestAncestor(tree) => tree.metric(),
        }
    }

    /// Returns the base value used by this tree.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustknn::{CoverTree, Distance};
    ///
    /// struct SimpleDistance;
    /// impl Distance<f64> for SimpleDistance {
    ///     fn distance(&self, p: &f64, q: &f64) -> f64 {
    ///         (p - q).abs()
    ///     }
    /// }
    ///
    /// let tree = CoverTree::new(SimpleDistance, 1.3);
    /// assert_eq!(tree.base_value(), 1.3);
    /// ```
    pub fn base_value(&self) -> f64 {
        match self {
            CoverTree::Simplified(tree) => tree.base_value(),
            CoverTree::NearestAncestor(tree) => tree.base_value(),
        }
    }

    /// Inserts a point into the cover tree.
    ///
    /// The behavior depends on the variant:
    /// - **Simplified**: Fast insertion without rebalancing (Algorithm 2)
    /// - **Nearest Ancestor**: Slower insertion with rebalancing (Algorithm 3)
    ///
    /// # Arguments
    ///
    /// * `point` - The point to insert (ownership transferred)
    ///
    /// # Time Complexity
    ///
    /// O(c^6 log n) where c is the doubling constant and n is the number of points.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustknn::{CoverTree, Distance};
    ///
    /// struct SimpleDistance;
    /// impl Distance<f64> for SimpleDistance {
    ///     fn distance(&self, p: &f64, q: &f64) -> f64 {
    ///         (p - q).abs()
    ///     }
    /// }
    ///
    /// let mut tree = CoverTree::new(SimpleDistance, 1.3);
    /// tree.insert(10.0);
    /// tree.insert(20.0);
    /// assert_eq!(tree.len(), 2);
    /// ```
    pub fn insert(&mut self, point: T) {
        match self {
            CoverTree::Simplified(tree) => tree.insert(point),
            CoverTree::NearestAncestor(tree) => tree.insert(point),
        }
    }

    /// Finds the nearest neighbor to a query point.
    ///
    /// This implements Algorithm 1 from the paper (nearest neighbor query with pruning).
    /// The Nearest Ancestor variant typically performs 10-30% fewer distance computations
    /// than the Simplified variant due to better tree structure.
    ///
    /// # Arguments
    ///
    /// * `query` - The query point (borrowed)
    ///
    /// # Returns
    ///
    /// * `Some(&T)` - A reference to the nearest point in the tree
    /// * `None` - If the tree is empty
    ///
    /// # Time Complexity
    ///
    /// O(c^6 log n) where c is the doubling constant and n is the number of points.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustknn::{CoverTree, Distance};
    ///
    /// struct SimpleDistance;
    /// impl Distance<f64> for SimpleDistance {
    ///     fn distance(&self, p: &f64, q: &f64) -> f64 {
    ///         (p - q).abs()
    ///     }
    /// }
    ///
    /// let mut tree = CoverTree::new(SimpleDistance, 1.3);
    /// tree.insert(10.0);
    /// tree.insert(20.0);
    ///
    /// let nearest = tree.find_nearest(&15.0);
    /// // Could be either 10.0 or 20.0 depending on tree structure
    /// assert!(nearest == Some(&10.0) || nearest == Some(&20.0));
    /// ```
    ///
    /// Example showing None for empty tree:
    ///
    /// ```rust
    /// use rustknn::{CoverTree, Distance};
    ///
    /// struct SimpleDistance;
    /// impl Distance<f64> for SimpleDistance {
    ///     fn distance(&self, p: &f64, q: &f64) -> f64 {
    ///         (p - q).abs()
    ///     }
    /// }
    ///
    /// let tree = CoverTree::new(SimpleDistance, 1.3);
    /// assert_eq!(tree.find_nearest(&15.0), None);
    /// ```
    pub fn find_nearest(&self, query: &T) -> Option<&T> {
        match self {
            CoverTree::Simplified(tree) => tree.find_nearest(query),
            CoverTree::NearestAncestor(tree) => tree.find_nearest(query),
        }
    }

    /// Find k nearest neighbors of a query point.
    ///
    /// Returns up to k nearest neighbors as `(point, distance)` tuples, sorted by distance (ascending).
    ///
    /// # Arguments
    ///
    /// * `query` - The query point to search for
    /// * `k` - Number of nearest neighbors to find
    ///
    /// # Returns
    ///
    /// * `Vec<(&T, f64)>` - Vector of up to k nearest neighbors as (point reference, distance) pairs
    ///   - Sorted by distance (closest first)
    ///   - Length ≤ min(k, tree.len())
    ///   - Returns empty vector if k=0 or tree is empty
    ///
    /// # Deduplication
    ///
    /// Algorithm-created duplicate nodes (from tree merging/level alignment) are automatically
    /// skipped. User's intentional duplicate values are correctly returned.
    ///
    /// # Time Complexity
    ///
    /// O(c^6 log n) where c is the doubling constant and n is the number of points.
    /// Same asymptotic complexity as single nearest neighbor search.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustknn::{CoverTree, Distance};
    ///
    /// struct SimpleDistance;
    /// impl Distance<f64> for SimpleDistance {
    ///     fn distance(&self, p: &f64, q: &f64) -> f64 {
    ///         (p - q).abs()
    ///     }
    /// }
    ///
    /// let mut tree = CoverTree::new(SimpleDistance, 1.3);
    /// tree.insert(1.0);
    /// tree.insert(5.0);
    /// tree.insert(10.0);
    /// tree.insert(15.0);
    ///
    /// // Find 3 nearest neighbors of 6.0
    /// let neighbors = tree.find_k_nearest(&6.0, 3);
    /// assert_eq!(neighbors.len(), 3);
    ///
    /// // Closest should be 5.0
    /// assert_eq!(*neighbors[0].0, 5.0);
    /// assert!((neighbors[0].1 - 1.0).abs() < 1e-10);
    /// ```
    pub fn find_k_nearest(&self, query: &T, k: usize) -> Vec<(&T, f64)> {
        match self {
            CoverTree::Simplified(tree) => tree.find_k_nearest(query, k),
            CoverTree::NearestAncestor(tree) => tree.find_k_nearest(query, k),
        }
    }

    /// Batch k-NN: builds a query tree internally and uses dual-tree traversal.
    ///
    /// For each query point, finds up to k nearest neighbors in this tree. Uses
    /// dual-tree traversal to prune entire subtree combinations, dramatically
    /// reducing distance computations compared to running single-tree k-NN for
    /// each query independently.
    ///
    /// # Arguments
    ///
    /// * `queries` - Slice of query points
    /// * `k` - Number of nearest neighbors to find per query point
    ///
    /// # Returns
    ///
    /// `Vec<Vec<(&T, f64)>>` — outer Vec has one entry per query point (same order
    /// as input), inner Vec contains up to k neighbors sorted by distance (ascending).
    ///
    /// # Examples
    ///
    /// ```
    /// use rustknn::{CoverTree, Distance};
    ///
    /// #[derive(Clone)]
    /// struct SimpleDistance;
    /// impl Distance<f64> for SimpleDistance {
    ///     fn distance(&self, p: &f64, q: &f64) -> f64 {
    ///         (p - q).abs()
    ///     }
    /// }
    ///
    /// let mut tree = CoverTree::new(SimpleDistance, 1.3);
    /// for i in 0..20 {
    ///     tree.insert(i as f64);
    /// }
    ///
    /// let queries = vec![5.5, 15.5];
    /// let results = tree.find_k_nearest_batch(&queries, 3);
    /// assert_eq!(results.len(), 2);
    /// ```
    pub fn find_k_nearest_batch(&self, queries: &[T], k: usize) -> Vec<Vec<(&T, f64)>>
    where
        D: Clone,
    {
        match self {
            CoverTree::Simplified(tree) => tree.find_k_nearest_batch(queries, k),
            CoverTree::NearestAncestor(tree) => tree.find_k_nearest_batch(queries, k),
        }
    }

    /// Same-set k-NN: find k nearest neighbors for each point in the tree itself.
    ///
    /// Self-matches are excluded (a point is not its own neighbor). Uses dual-tree
    /// traversal with the same tree as both query and reference.
    ///
    /// # Arguments
    ///
    /// * `k` - Number of nearest neighbors to find per point
    ///
    /// # Returns
    ///
    /// `Vec<Vec<(&T, f64)>>` — one entry per non-duplicate point in the tree
    /// (DFS order), each containing up to k neighbors sorted by distance.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustknn::{CoverTree, Distance};
    ///
    /// struct SimpleDistance;
    /// impl Distance<f64> for SimpleDistance {
    ///     fn distance(&self, p: &f64, q: &f64) -> f64 {
    ///         (p - q).abs()
    ///     }
    /// }
    ///
    /// let mut tree = CoverTree::new(SimpleDistance, 1.3);
    /// tree.insert(0.0);
    /// tree.insert(1.0);
    /// tree.insert(3.0);
    ///
    /// let results = tree.find_k_nearest_self(1);
    /// assert_eq!(results.len(), 3);
    /// ```
    pub fn find_k_nearest_self(&self, k: usize) -> Vec<Vec<(&T, f64)>> {
        match self {
            CoverTree::Simplified(tree) => tree.find_k_nearest_self(k),
            CoverTree::NearestAncestor(tree) => tree.find_k_nearest_self(k),
        }
    }

    /// Recomputes exact maxdist for all nodes in the tree.
    ///
    /// During insertion, both Simplified and Nearest Ancestor variants maintain maxdist as
    /// **upper bounds** using the triangle inequality. This is efficient and safe for correctness,
    /// but may result in suboptimal query pruning.
    ///
    /// This method performs a complete tree traversal to compute **exact** maxdist values:
    /// `maxdist(p) = max{d(p,q) : q ∈ descendants(p)}`. Exact bounds enable optimal query
    /// pruning, typically reducing distance computations by 10-30% (per the paper).
    ///
    /// # Variant-Specific Behavior
    ///
    /// - **Simplified**: Recomputes all maxdist values (recommended after batch insertions)
    /// - **NearestAncestor**: No-op (already maintains exact maxdist during operations)
    ///
    /// # When to Call
    ///
    /// - **After batch insertions**: Insert many points, then call this once
    /// - **Before query-heavy workloads**: Tighten bounds before performing many queries
    /// - **Recommended for Simplified variant only**: NACoverTree doesn't need this
    ///
    /// # Performance
    ///
    /// - Time complexity: O(n²) worst case (examines all ancestor-descendant pairs)
    /// - In practice: Much faster due to branch-and-bound pruning
    /// - Cost is amortized over subsequent queries
    ///
    /// # Matches HLearn
    ///
    /// This corresponds to HLearn's `setMaxDescendentDistance` function.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustknn::{CoverTree, Distance};
    ///
    /// struct SimpleDistance;
    /// impl Distance<f64> for SimpleDistance {
    ///     fn distance(&self, p: &f64, q: &f64) -> f64 {
    ///         (p - q).abs()
    ///     }
    /// }
    ///
    /// let mut tree = CoverTree::new(SimpleDistance, 1.3);
    ///
    /// // Insert many points
    /// for i in 0..1000 {
    ///     tree.insert(i as f64);
    /// }
    ///
    /// // Tighten maxdist bounds for optimal query performance
    /// tree.recompute_maxdist();
    ///
    /// // Now queries will benefit from tighter pruning
    /// let nearest = tree.find_nearest(&500.5);
    /// ```
    pub fn recompute_maxdist(&mut self) {
        match self {
            CoverTree::Simplified(tree) => tree.recompute_maxdist(),
            CoverTree::NearestAncestor(tree) => tree.recompute_maxdist(),
        }
    }

    /// Merge two cover trees, consuming both and returning a merged tree.
    ///
    /// This implements Algorithm 4 from the paper (tree merging). The merge operation
    /// combines two trees while maintaining all three cover tree invariants.
    ///
    /// **Current Limitation**: Only SimplifiedCoverTree variants can be merged.
    /// NAOptimizedCoverTree merge is deferred to future work.
    ///
    /// # Arguments
    ///
    /// * `other` - The tree to merge with this one
    ///
    /// # Returns
    ///
    /// A new `CoverTree` containing all points from both trees. If merging NearestAncestor variants,
    /// returns a Simplified variant (merging doesn't maintain nearest ancestor invariant).
    ///
    /// # Panics
    ///
    /// * Panics if the two trees have different base values
    /// * Panics if the two trees are different variants (cannot merge Simplified with NearestAncestor)
    /// * Panics if the trees' levels are too far apart to align safely (see
    ///   [`SimplifiedCoverTree::merge`])
    ///
    /// # Time Complexity
    ///
    /// O(n + m) where n and m are the sizes of the two trees.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustknn::{CoverTree, Distance};
    ///
    /// struct SimpleDistance;
    /// impl Distance<f64> for SimpleDistance {
    ///     fn distance(&self, p: &f64, q: &f64) -> f64 {
    ///         (p - q).abs()
    ///     }
    /// }
    ///
    /// let mut tree1 = CoverTree::new(SimpleDistance, 1.3);
    /// tree1.insert(1.0);
    /// tree1.insert(2.0);
    ///
    /// let mut tree2 = CoverTree::new(SimpleDistance, 1.3);
    /// tree2.insert(10.0);
    /// tree2.insert(20.0);
    ///
    /// let merged = tree1.merge(tree2);
    /// assert_eq!(merged.len(), 4);
    /// ```
    pub fn merge(self, other: Self) -> Self {
        match (self, other) {
            (CoverTree::Simplified(t1), CoverTree::Simplified(t2)) => {
                CoverTree::Simplified(t1.merge(t2))
            }
            (CoverTree::NearestAncestor(t1), CoverTree::NearestAncestor(t2)) => {
                // NACoverTree.merge returns SimplifiedCoverTree (see its documentation)
                CoverTree::Simplified(t1.merge(t2))
            }
            _ => panic!("Cannot merge trees of different variants (Simplified and NearestAncestor)"),
        }
    }
}
