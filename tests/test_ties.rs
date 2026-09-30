#[test]
fn test_query_0_5_0_5() {
    use rustknn::simplified::SimplifiedCoverTree;
    use rustknn::distance::Distance;

    #[derive(Clone, Debug, PartialEq)]
    struct Point2D { x: f64, y: f64 }

    struct EuclideanDistance2D;
    impl Distance<Point2D> for EuclideanDistance2D {
        fn distance(&self, p: &Point2D, q: &Point2D) -> f64 {
            let dx = p.x - q.x;
            let dy = p.y - q.y;
            (dx * dx + dy * dy).sqrt()
        }
    }

    let mut tree = SimplifiedCoverTree::new(EuclideanDistance2D, 1.3);

    tree.insert(Point2D { x: 0.0, y: 0.0 });
    tree.insert(Point2D { x: 1.0, y: 0.0 });
    tree.insert(Point2D { x: 0.0, y: 1.0 });

    let packed = tree.pack();

    // All three points are at distance sqrt(0.5) from the query, so any of
    // them is a correct nearest neighbor.
    let query = Point2D { x: 0.5, y: 0.5 };
    let expected = 0.5_f64.sqrt();

    let result = packed.find_nearest(&query).expect("tree is non-empty");
    let d = EuclideanDistance2D.distance(result, &query);
    assert!((d - expected).abs() < 1e-12, "nearest distance {} != {}", d, expected);

    // k=3 must return all three equidistant points, each exactly once.
    let knn = packed.find_k_nearest(&query, 3);
    assert_eq!(knn.len(), 3);
    for (_, dist) in &knn {
        assert!((dist - expected).abs() < 1e-12);
    }
    for i in 0..knn.len() {
        for j in (i + 1)..knn.len() {
            assert_ne!(knn[i].0, knn[j].0, "duplicate point in k-NN result");
        }
    }
}
