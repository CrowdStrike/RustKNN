# RustKNN

Cover trees for exact K-nearest-neighbor search in Rust.

RustKNN is the implementation accompanying the NeurIPS 2026 paper *KNN
Implementation Details Can Dramatically Change Performance: An Example from Cover
Trees*. It provides three cover tree construction variants and three batch query
modes in a single codebase, so that each implementation choice can be measured in
isolation.

## Features

- **Three construction variants**
  - **Simplified** cover tree: exactly one node per point, no rebalancing (Izbicki & Shelton, 2015).
  - **Nearest Ancestor** cover tree: rebalances on insert to maintain the nearest-ancestor invariant.
  - **Packed** cover tree: a simplified tree repacked into a contiguous, depth-first array for cache locality.
- **Three batch query modes**
  - **Dual-tree**: builds a query tree and traverses it against the reference tree.
  - **Single-tree**: one independent reference-tree search per query point.
  - **Batch single-tree**: walks the query tree but does single-tree reference descent at each
    scale, after Beygelzimer, Kakade & Langford (2006).
- Generic over point type and metric through the `Distance` trait. Built-in `EuclideanDistance`
  and `ManhattanDistance` for `Vec<f64>`, with an early-exit `distance_with_bound`.
- Configurable base (default 1.3).
- Tree merging and parallel construction (Rayon).
- Reference baselines on the same `Distance` and candidate-tracking code: a median-split
  KD-tree (`rustknn::kdtree`) and a brute-force scan (`rustknn::naive`).

## Installation

RustKNN is not published on crates.io. Add it as a git dependency:

```toml
[dependencies]
rustknn = { git = "https://github.com/CrowdStrike/RustKNN" }
```

Build with `--release` for meaningful performance. The release profile enables LTO and
`codegen-units = 1`. For best performance on your own machine, also set
`RUSTFLAGS="-C target-cpu=native"`.

## Quick start

### Nearest neighbor with a custom metric

```rust
use rustknn::{CoverTree, Distance};

#[derive(Clone, Debug)]
struct Point2D { x: f64, y: f64 }

struct Euclidean;
impl Distance<Point2D> for Euclidean {
    fn distance(&self, p: &Point2D, q: &Point2D) -> f64 {
        ((p.x - q.x).powi(2) + (p.y - q.y).powi(2)).sqrt()
    }
}

fn main() {
    let mut tree = CoverTree::new(Euclidean, 1.3);
    tree.insert(Point2D { x: 0.0, y: 0.0 });
    tree.insert(Point2D { x: 1.0, y: 1.0 });
    tree.insert(Point2D { x: 2.0, y: 2.0 });

    let query = Point2D { x: 1.4, y: 1.4 };
    println!("nearest: {:?}", tree.find_nearest(&query));

    // (&point, distance) pairs, closest first
    for (p, d) in tree.find_k_nearest(&query, 2) {
        println!("  {:?} at {:.3}", p, d);
    }
}
```

### Packed tree and batch queries on `Vec<f64>`

```rust
use rustknn::{EuclideanDistance, SimplifiedCoverTree};

fn main() {
    let points: Vec<Vec<f64>> = (0..1000)
        .map(|i| vec![(i % 37) as f64, (i % 91) as f64, (i % 13) as f64])
        .collect();
    let queries: Vec<Vec<f64>> = vec![vec![1.0, 2.0, 3.0], vec![10.0, 20.0, 5.0]];

    let mut tree = SimplifiedCoverTree::new(EuclideanDistance, 1.3);
    for p in points {
        tree.insert(p);
    }
    let packed = tree.pack();

    // Single-tree: one search per query.
    let single: Vec<_> = queries.iter().map(|q| packed.find_k_nearest(q, 5)).collect();

    // Dual-tree: one result list per query, in the same order as `queries`.
    let dual = packed.find_k_nearest_batch(&queries, 5);

    assert_eq!(single.len(), dual.len());
}
```

## Choosing a variant and query mode

| | Simplified | Nearest Ancestor | Packed |
|---|---|---|---|
| Construction | Fastest | Slower (rebalancing) | `SimplifiedCoverTree::pack()`, or `NACoverTree::into_simplified().pack()` |
| Queries | Good | Fewer distance computations | Fastest (cache-friendly layout) |
| Merging | `merge()` | Returns a Simplified tree | No (read-only) |
| Best for | Large or parallel builds | Query-heavy workloads | Repeated queries on static data |

The paper's main finding is that the best query mode depends on the workload.
Picking the wrong one can cost orders of magnitude in runtime. The API for each
mode:

| Mode | Entry points |
|---|---|
| Single-tree | `find_nearest`, `find_k_nearest` (loop over queries) |
| Dual-tree | `find_k_nearest_batch` (held-out queries), `find_k_nearest_self` (all-NN) |
| Batch single-tree | `find_k_nearest_batch_single_self` (all-NN, on `SimplifiedCoverTree` and `PackedCoverTree`), `PackedCoverTree::find_k_nearest_batch_single` (held-out queries) |

## Input limits

These limits turn inputs that would otherwise hang or crash the process into ordinary
(catchable) panics:

- **Base:** every constructor requires a finite base of at least `rustknn::MIN_BASE`
  (1.1). The default, 1.3, is the usual choice.
- **Tree depth:** recursive operations (queries, construction, merging, packing) may use
  at most `rustknn::RECURSION_STACK_BUDGET` (1 MiB) of stack per thread, and panic beyond
  it rather than overflowing the stack. Only data spanning hundreds of orders of magnitude
  gets close: in release builds, the deepest tree in the paper's benchmark datasets (covtype
  built in parallel at base 1.1, 153 levels) runs every query mode in under half the
  budget. Debug builds have larger stack frames, so they reach the limit sooner.
  If a nearest-ancestor insert hits the limit, the tree is left empty.
- **`k`:** any `k` is accepted, including `usize::MAX`. Memory grows with the number of
  neighbors actually found, not with `k`.

## Cargo features

All features are off by default.

| Feature | Effect |
|---|---|
| `instrument` | Counts distance computations (`reset_distance_count` / `get_distance_count`). Adds a small per-call overhead. |
| `no-early-exit` | Disables the partial-distance early exit in `distance_with_bound`. |
| `no-halfsort` | Uses a full sort instead of a partial (half) sort of candidate lists. |
| `no-smallvec` | Uses `Vec` instead of `SmallVec` in hot query loops. |
| `no-k1-special` | Disables the k=1 fast path in candidate tracking. |
| `no-triangle-filter` | Disables triangle-inequality pre-filters in traversal loops. |
| `hashmap-batch` | Uses a `HashMap` instead of a sorted `Vec` for batch result mapping. |

The `no-*` and `hashmap-batch` features are ablation switches. Each one turns off a
single micro-optimization so its effect can be measured, as in the paper's ablation
study. Do not enable them in normal use.

## Project structure

```text
src/
├── lib.rs                 Public API and re-exports
├── distance.rs            Distance trait; Euclidean and Manhattan kernels
├── node.rs                Pointer-based tree node
├── tree.rs                CoverTree enum over Simplified / Nearest Ancestor
├── knn.rs                 k-NN candidate tracking (k=1 fast path, sorted array for k>1)
├── core/
│   ├── query.rs           Nearest-neighbor search
│   ├── knn_impl.rs        Single-tree k-NN
│   ├── batch_single.rs    Batch single-tree k-NN
│   └── dual_tree/         Dual-tree traversal
├── simplified/            Simplified cover tree: insert, merge
├── nearest_ancestor/      Nearest Ancestor cover tree: insert, rebalance
├── packed/                Packed layout and packed dual-tree traversals
├── parallel/              Parallel construction and merge-all
├── kdtree/                KD-tree reference baseline
└── naive.rs               Brute-force reference baseline
tests/                     Integration tests (synthetic data only)
examples/                  Runnable examples
```

## Building, testing, and examples

```bash
cargo build --release
cargo test
cargo test --all-features
cargo run --example euclidean_2d
cargo run --example knn_search
cargo run --release --example performance_comparison
```

## Scope of this repository

This repository contains the RustKNN library, its tests, and examples. It does not
include the benchmark harness, the third-party baselines, the experiment results, or
the datasets used in the paper. The experimental setup, datasets (UCI Machine Learning
Repository and MNIST), and baselines are described and cited in the paper.

## Citation

If you use RustKNN, please cite:

```bibtex
@inproceedings{khanna2026knn,
  title     = {{KNN} Implementation Details Can Dramatically Change Performance:
               An Example from Cover Trees},
  author    = {Khanna, Amol and Raff, Edward},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## References

- A. Beygelzimer, S. Kakade, and J. Langford. Cover trees for nearest neighbor. *ICML*, 2006.
- M. Izbicki and C. R. Shelton. Faster cover trees. *ICML*, 2015.

## License

MIT. See [LICENSE](LICENSE).
