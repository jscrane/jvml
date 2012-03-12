(ns ml.test-kmeans
  (:use (clojure test)
        (incanter core)
        (ml matlab kmeans testutil)))

(def X (:X (read-dataset-mat5 "data/ex7data2.mat")))

(def centroids (matrix [[3 3] [6 2] [8 5]]))

(deftest test-find-closest-centroids
  (is (= 0 (find-closest-centroid (trans [1.8421 4.6076]) centroids)))
  (is (= 2 (find-closest-centroid (trans [5.6586 4.8000]) centroids)))
  (is (= 1 (find-closest-centroid (trans [6.3526 3.2909]) centroids)))
  (is (= [0 2 1] (find-closest-centroids (matrix (take 3 X)) centroids))))

(deftest test-compute-centroids
  (let [approx (approximately 1e-6)
        idx (find-closest-centroids X centroids)
        new-centroids (map matrix (compute-centroids X idx 3))]
    (is (approx (matrix [[2.428301 3.157924] [5.813503 2.633656] [7.119387 3.616684]])
          new-centroids))))

(deftest test-run-kmeans
  (let [approx (approximately 1e-4)
        centroids (run-kmeans X centroids 10)]
    (is (approx (matrix [[1.9540 5.0256] [3.0437 1.0154] [6.0337 3.0005]]) centroids))))
