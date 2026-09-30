//! PackedNode - Index-based node for cache-friendly layout
//!
//! Unlike the pointer-based Node, PackedNode uses indices to reference children.
//! This enables contiguous memory layout and dramatically improves cache performance.

/// A node in a packed cover tree with index-based children
///
/// Uses a two-array approach for cache efficiency:
/// - All nodes stored in contiguous `nodes` array (depth-first order)
/// - Child indices stored in separate `child_indices` array (shared by all nodes)
///
/// This avoids storing Vec<usize> per node, keeping nodes small (~40 bytes)
/// so more nodes fit in L1/L2 cache.
///
/// # Memory Layout
///
/// ```text
/// nodes array (depth-first):
/// [0] Parent
/// [1] Child1
/// [2] Grandchild1a
/// [3] Grandchild1b
/// [4] Child2
/// [5] Grandchild2a
///
/// child_indices array (shared):
/// [0] 2  ← Child1's children start here
/// [1] 3
/// [2] 1  ← Parent's children start here
/// [3] 4
/// [4] 5  ← Child2's children start here
///
/// Parent.child_index_start=2, count=2 → uses child_indices[2..4] = [1, 4]
/// Child1.child_index_start=0, count=2 → uses child_indices[0..2] = [2, 3]
/// Child2.child_index_start=4, count=1 → uses child_indices[4..5] = [5]
/// ```
///
/// # Cache Efficiency
///
/// - Nodes stay ~32 bytes (vs ~96 bytes with Vec per node)
/// - ~1000 nodes fit in 32KB L1 cache (vs ~340 with Vec approach)
/// - Child indices accessed sequentially (cache-friendly)
/// - Depth-first layout keeps related nodes close in memory
#[derive(Clone, Debug)]
pub struct PackedNode<T: Clone> {
    /// The data point stored at this node
    pub point: T,

    /// The level of this node in the tree
    pub level: i32,

    /// Maximum distance from this node to any descendant
    ///
    /// Used for query pruning (same semantics as unpacked Node)
    pub maxdist: f64,

    /// Distance from parent's point to this node's point: d(parent.point, self.point).
    ///
    /// Computed during the packing process. For the root node, this is 0.0
    /// (no parent). This avoids recomputing parent-child distances during
    /// dual-tree traversal (Phase 1 triangle-inequality pre-pruning).
    pub d_parent: f64,

    /// Index into shared child_indices array where this node's children start
    ///
    /// This node's children are at: child_indices[child_index_start..child_index_start+child_index_count]
    /// Stored as u32 (4 bytes) instead of usize (8 bytes) to reduce node size by 8 bytes,
    /// fitting more nodes per cache line. u32 supports up to ~4 billion entries.
    pub child_index_start: u32,

    /// Number of children this node has
    ///
    /// Stored as u32 to match child_index_start and reduce node size.
    pub child_index_count: u32,

    /// Internal flag marking algorithm-created duplicates
    ///
    /// When true, this node is skipped during k-NN search to avoid returning
    /// duplicate results from tree merging operations. User's intentional
    /// duplicate values have this flag set to false.
    ///
    /// This flag is copied from the unpacked Node during the packing process.
    pub is_duplicate: bool,
}

impl<T: Clone> PackedNode<T> {
    /// Calculate the covering distance for this node
    ///
    /// covdist(p) = base^level(p)
    ///
    /// Children must be within this distance from parent
    #[inline]
    #[allow(dead_code)]
    pub fn covdist(&self, base: f64) -> f64 {
        base.powi(self.level)
    }

    /// Calculate the separating distance for this node
    ///
    /// sepdist(p) = base^(level(p) - 1)
    ///
    /// Children must be separated by more than this distance from each other
    #[inline]
    #[allow(dead_code)]
    pub fn sepdist(&self, base: f64) -> f64 {
        base.powi(self.level - 1)
    }

    /// Get range for child indices in the shared child_indices array
    ///
    /// Returns range [child_index_start..child_index_start + child_index_count]
    /// which can be used to slice the child_indices array to get this node's children.
    /// Casts from u32 to usize for array indexing.
    #[inline]
    pub fn child_index_range(&self) -> std::ops::Range<usize> {
        let start = self.child_index_start as usize;
        let count = self.child_index_count as usize;
        start..start + count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packed_node_covdist() {
        let node = PackedNode {
            point: 0.0,
            level: 3,
            maxdist: 10.0,
            d_parent: 0.0,
            child_index_start: 0,
            child_index_count: 0,
            is_duplicate: false,
        };

        assert!((node.covdist(1.3) - 1.3_f64.powi(3)).abs() < 1e-10);
    }

    #[test]
    fn test_packed_node_sepdist() {
        let node = PackedNode {
            point: 0.0,
            level: 3,
            maxdist: 10.0,
            d_parent: 0.0,
            child_index_start: 0,
            child_index_count: 0,
            is_duplicate: false,
        };

        assert!((node.sepdist(1.3) - 1.3_f64.powi(2)).abs() < 1e-10);
    }

    #[test]
    fn test_child_index_range_empty() {
        let node = PackedNode {
            point: 0.0,
            level: 3,
            maxdist: 10.0,
            d_parent: 0.0,
            child_index_start: 5,
            child_index_count: 0,
            is_duplicate: false,
        };

        let range = node.child_index_range();
        assert_eq!(range.len(), 0);
        assert_eq!(range.start, 5);
        assert_eq!(range.end, 5);
    }

    #[test]
    fn test_packed_node_layout() {
        use std::mem;

        println!("PackedNode<f64> size: {}", mem::size_of::<PackedNode<f64>>());
        println!("PackedNode<f64> align: {}", mem::align_of::<PackedNode<f64>>());
        println!("  point offset:             {}", mem::offset_of!(PackedNode<f64>, point));
        println!("  level offset:             {}", mem::offset_of!(PackedNode<f64>, level));
        println!("  maxdist offset:           {}", mem::offset_of!(PackedNode<f64>, maxdist));
        println!("  d_parent offset:          {}", mem::offset_of!(PackedNode<f64>, d_parent));
        println!("  child_index_start offset: {}", mem::offset_of!(PackedNode<f64>, child_index_start));
        println!("  child_index_count offset: {}", mem::offset_of!(PackedNode<f64>, child_index_count));
        println!("  is_duplicate offset:      {}", mem::offset_of!(PackedNode<f64>, is_duplicate));

        // Assert hot fields (point, maxdist, d_parent) span at most 32 bytes,
        // fitting comfortably within a single 64-byte cache line.
        let point_off = mem::offset_of!(PackedNode<f64>, point);
        let maxdist_off = mem::offset_of!(PackedNode<f64>, maxdist);
        let d_parent_off = mem::offset_of!(PackedNode<f64>, d_parent);
        let hot_start = point_off.min(maxdist_off).min(d_parent_off);
        let hot_end = (point_off + 8).max(maxdist_off + 8).max(d_parent_off + 8);
        assert!(
            hot_end - hot_start <= 32,
            "Hot fields span {} bytes (want <= 32). Consider reordering PackedNode fields.",
            hot_end - hot_start,
        );
    }

    #[test]
    fn test_child_index_range_multiple() {
        let node = PackedNode {
            point: 0.0,
            level: 3,
            maxdist: 10.0,
            d_parent: 0.0,
            child_index_start: 10,
            child_index_count: 3,
            is_duplicate: false,
        };

        let range = node.child_index_range();
        let indices: Vec<usize> = range.collect();
        assert_eq!(indices, vec![10, 11, 12]);
    }
}
