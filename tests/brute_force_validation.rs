//! Randomized checks of every query mode against brute force, on trees built every
//! supported way, plus structural checks of the invariants the queries rely on.

use rustknn::kdtree::KdTree;
use rustknn::naive::NaiveNN;
use rustknn::parallel::{build_parallel, ParallelVariant};
use rustknn::{CoverTree, Distance, EuclideanDistance, NACoverTree, Node, SimplifiedCoverTree};

type P = Vec<f64>;

/// Deterministic xorshift generator in [0, 1).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Points clustered on a grid (many near-ties), plus `dups` exact copies.
fn dataset(seed: u64, n: usize, dim: usize, dups: usize) -> Vec<P> {
    let mut r = Rng(seed);
    let mut pts: Vec<P> = (0..n)
        .map(|_| (0..dim).map(|_| (r.next() * 10.0).round() / 10.0 + r.next() * 1e-3).collect())
        .collect();
    for i in 0..dups {
        let p = pts[i % 5].clone();
        pts.push(p);
    }
    pts
}

fn queries(seed: u64, n: usize, dim: usize) -> Vec<P> {
    let mut r = Rng(seed);
    (0..n).map(|_| (0..dim).map(|_| r.next() * 1.2 - 0.1).collect()).collect()
}

/// Distances rounded to 1e-9 so equivalent results compare equal.
fn key(d: impl IntoIterator<Item = f64>) -> Vec<i64> {
    let mut v: Vec<f64> = d.into_iter().collect();
    v.sort_by(f64::total_cmp);
    v.into_iter().map(|x| (x * 1e9).round() as i64).collect()
}

fn brute(pts: &[P], q: &P, k: usize, skip: Option<usize>) -> Vec<i64> {
    let mut d: Vec<f64> = pts
        .iter()
        .enumerate()
        .filter(|(i, _)| Some(*i) != skip)
        .map(|(_, p)| EuclideanDistance.distance(p, q))
        .collect();
    d.sort_by(f64::total_cmp);
    d.truncate(k);
    key(d)
}

fn rows<'a>(rs: impl IntoIterator<Item = &'a [(&'a P, f64)]>) -> Vec<Vec<i64>> {
    rs.into_iter().map(|r| key(r.iter().map(|x| x.1))).collect()
}

fn sorted(mut v: Vec<Vec<i64>>) -> Vec<Vec<i64>> {
    v.sort();
    v
}

/// Every supported way of building a simplified tree over `pts`.
fn builders(pts: &[P], base: f64) -> Vec<(&'static str, SimplifiedCoverTree<P, EuclideanDistance>)> {
    let insert = |ps: &[P]| {
        let mut t = SimplifiedCoverTree::new(EuclideanDistance, base);
        for p in ps {
            t.insert(p.clone());
        }
        t
    };
    let mut na = NACoverTree::new(EuclideanDistance, base);
    for p in pts {
        na.insert(p.clone());
    }
    let (a, b) = pts.split_at(pts.len() / 3);
    let insert_na = |ps: &[P]| {
        let mut t = NACoverTree::new(EuclideanDistance, base);
        for p in ps {
            t.insert(p.clone());
        }
        t
    };
    vec![
        ("insert", insert(pts)),
        ("nearest-ancestor", na.into_simplified()),
        ("merge", insert(a).merge(insert(b))),
        ("na-merge", insert_na(a).merge(insert_na(b))),
        ("parallel", build_parallel(EuclideanDistance, base, pts.to_vec(), Some(5), ParallelVariant::Simplified)),
        ("parallel-na", build_parallel(EuclideanDistance, base, pts.to_vec(), Some(5), ParallelVariant::NearestAncestor)),
    ]
}

const CASES: [(u64, usize, usize, usize); 4] = [
    (1, 200, 2, 0),
    (2, 160, 8, 0),
    (3, 120, 3, 40), // exact duplicates
    (4, 64, 1, 0),   // collinear: the triangle inequality is often tight
];
const BASES: [f64; 3] = [1.1, 1.3, 2.0];
const KS: [usize; 4] = [1, 2, 5, 17];

#[test]
fn single_and_dual_tree_queries_match_brute_force() {
    for &(seed, n, dim, dups) in &CASES {
        let pts = dataset(seed, n, dim, dups);
        let qs = queries(seed + 100, 40, dim);
        for &base in &BASES {
            let mut na = NACoverTree::new(EuclideanDistance, base);
            let mut ct = CoverTree::new_nearest_ancestor(EuclideanDistance, base);
            for p in &pts {
                na.insert(p.clone());
                ct.insert(p.clone());
            }
            let kd = KdTree::new(pts.clone(), EuclideanDistance);
            let naive = NaiveNN::new(pts.clone(), EuclideanDistance);
            for &k in &KS {
                let want: Vec<Vec<i64>> = qs.iter().map(|q| brute(&pts, q, k, None)).collect();
                let ctx = format!("seed={seed} base={base} k={k}");
                for (name, t) in builders(&pts, base) {
                    let single: Vec<_> = qs.iter().map(|q| t.find_k_nearest(q, k)).collect();
                    assert_eq!(rows(single.iter().map(Vec::as_slice)), want, "{ctx} {name} single");
                    let dual = t.find_k_nearest_batch(&qs, k);
                    assert_eq!(rows(dual.iter().map(Vec::as_slice)), want, "{ctx} {name} dual");
                    let packed = t.pack();
                    let single: Vec<_> = qs.iter().map(|q| packed.find_k_nearest(q, k)).collect();
                    assert_eq!(rows(single.iter().map(Vec::as_slice)), want, "{ctx} {name} packed single");
                    let dual = packed.find_k_nearest_batch(&qs, k);
                    assert_eq!(rows(dual.iter().map(Vec::as_slice)), want, "{ctx} {name} packed dual");
                }
                let single: Vec<_> = qs.iter().map(|q| na.find_k_nearest(q, k)).collect();
                assert_eq!(rows(single.iter().map(Vec::as_slice)), want, "{ctx} NA single");
                let dual = na.find_k_nearest_batch(&qs, k);
                assert_eq!(rows(dual.iter().map(Vec::as_slice)), want, "{ctx} NA dual");
                let single: Vec<_> = qs.iter().map(|q| ct.find_k_nearest(q, k)).collect();
                assert_eq!(rows(single.iter().map(Vec::as_slice)), want, "{ctx} CoverTree single");
                let single: Vec<_> = qs.iter().map(|q| kd.find_k_nearest(q, k)).collect();
                assert_eq!(rows(single.iter().map(Vec::as_slice)), want, "{ctx} KdTree");
                let single: Vec<_> = qs.iter().map(|q| naive.find_k_nearest(q, k)).collect();
                assert_eq!(rows(single.iter().map(Vec::as_slice)), want, "{ctx} NaiveNN");
            }
        }
    }
}

#[test]
fn self_queries_match_brute_force() {
    for &(seed, n, dim, dups) in &CASES {
        let pts = dataset(seed, n, dim, dups);
        for &base in &BASES {
            for &k in &KS {
                let want = sorted((0..pts.len()).map(|i| brute(&pts, &pts[i], k, Some(i))).collect());
                let ctx = format!("seed={seed} base={base} k={k}");
                for (name, t) in builders(&pts, base) {
                    let dual = t.find_k_nearest_self(k);
                    assert_eq!(sorted(rows(dual.iter().map(Vec::as_slice))), want, "{ctx} {name} dual self");
                    let bs = t.find_k_nearest_batch_single_self(k);
                    assert_eq!(sorted(rows(bs.iter().map(|(_, r)| r.as_slice()))), want, "{ctx} {name} batch single self");
                    let packed = t.pack();
                    let dual = packed.find_k_nearest_self(k);
                    assert_eq!(sorted(rows(dual.iter().map(Vec::as_slice))), want, "{ctx} {name} packed dual self");
                    let bs = packed.find_k_nearest_batch_single_self(k);
                    assert_eq!(sorted(rows(bs.iter().map(|(_, r)| r.as_slice()))), want, "{ctx} {name} packed batch single self");
                }
            }
        }
    }
}

#[test]
fn packed_batch_single_held_out_matches_brute_force() {
    for &(seed, n, dim, dups) in &CASES {
        let pts = dataset(seed, n, dim, dups);
        let qs = queries(seed + 200, 60, dim);
        for &base in &BASES {
            for &k in &KS {
                let want: Vec<Vec<i64>> = qs.iter().map(|q| brute(&pts, q, k, None)).collect();
                for (name, t) in builders(&pts, base) {
                    let packed = t.pack();
                    let got = packed.find_k_nearest_batch_single(&qs, k);
                    assert_eq!(rows(got.iter().map(Vec::as_slice)), want, "seed={seed} base={base} k={k} {name}");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Structural invariants
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Shape {
    real_nodes: usize,
    depth: usize,
}

/// Checks that every node's `maxdist` bounds its subtree and every cached `d_parent`
/// is accurate (0.0 only for exact copies), and measures the tree.
fn check_shape(root: &Node<P>, what: &str) -> Shape {
    fn farthest(n: &Node<P>, from: &P) -> f64 {
        n.children
            .iter()
            .map(|c| EuclideanDistance.distance(from, &c.point).max(farthest(c, from)))
            .fold(0.0, f64::max)
    }
    let mut shape = Shape::default();
    let mut stack = vec![(root, 1)];
    while let Some((n, depth)) = stack.pop() {
        shape.depth = shape.depth.max(depth);
        if !n.is_duplicate {
            shape.real_nodes += 1;
        }
        let radius = farthest(n, &n.point);
        assert!(radius <= n.maxdist + 1e-12, "{what}: maxdist {} < subtree radius {}", n.maxdist, radius);
        for c in &n.children {
            let d = EuclideanDistance.distance(&n.point, &c.point);
            if c.d_parent == 0.0 {
                assert_eq!(d, 0.0, "{what}: d_parent is 0 for a child at distance {d}");
            } else {
                assert!((c.d_parent - d).abs() <= 1e-12, "{what}: stale d_parent {} != {}", c.d_parent, d);
            }
            stack.push((c, depth + 1));
        }
    }
    shape
}

#[test]
fn trees_keep_maxdist_and_d_parent_valid() {
    for &(seed, n, dim, dups) in &CASES {
        let pts = dataset(seed, n, dim, dups);
        for &base in &BASES {
            for (name, t) in builders(&pts, base) {
                let what = format!("seed={seed} base={base} {name}");
                let shape = check_shape(t.root_node().unwrap(), &what);
                assert_eq!(shape.real_nodes, pts.len(), "{what}: one real node per point");
            }
            let mut na = NACoverTree::new(EuclideanDistance, base);
            for p in &pts {
                na.insert(p.clone());
            }
            check_shape(na.root_node().unwrap(), &format!("seed={seed} base={base} NA"));
        }
    }
}

/// Exact duplicates used to chain one level deeper per copy, so a few thousand copies
/// overflowed the stack in recursive queries.
#[test]
fn exact_duplicates_keep_the_tree_shallow() {
    std::thread::Builder::new()
        .stack_size(2 << 20)
        .spawn(|| {
            let mut pts: Vec<P> = vec![vec![0.5, 0.5]; 5000];
            pts.extend((0..100).map(|i| vec![(i % 10) as f64 * 0.1, (i / 10) as f64 * 0.1]));
            for (name, t) in builders(&pts, 1.3) {
                let shape = check_shape(t.root_node().unwrap(), name);
                assert!(shape.depth <= 30, "{name}: depth {} with 5000 copies", shape.depth);
                assert_eq!(t.find_k_nearest(&vec![0.5, 0.5], 3).len(), 3);
            }
            let mut na = NACoverTree::new(EuclideanDistance, 1.3);
            for p in &pts {
                na.insert(p.clone());
            }
            let shape = check_shape(na.root_node().unwrap(), "NA");
            assert!(shape.depth <= 30, "NA: depth {} with 5000 copies", shape.depth);

            // A smaller group exercises the self queries, which visit every tied copy.
            let mut small: Vec<P> = vec![vec![0.5, 0.5]; 800];
            small.extend((0..50).map(|i| vec![i as f64 * 0.01, 0.0]));
            let mut t = SimplifiedCoverTree::new(EuclideanDistance, 1.3);
            for p in &small {
                t.insert(p.clone());
            }
            assert_eq!(t.find_k_nearest_self(3).len(), small.len());
            assert_eq!(t.find_k_nearest_batch_single_self(3).len(), small.len());
            let packed = t.pack();
            assert_eq!(packed.find_k_nearest_self(3).len(), small.len());
            assert_eq!(packed.find_k_nearest_batch_single_self(3).len(), small.len());
        })
        .unwrap()
        .join()
        .unwrap();
}
