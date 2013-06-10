(ns ml.test-ex7
  (:use (clojure test)
        (incanter core)
        [ml.ex7 :only (init-ex7)]
        [ml.util :only (feature-normalize)]
        (ml matlab kmeans pca testutil)))

(deftest test-find-closest-centroids
  (let [{:keys [centroids X]} (init-ex7)]
    (is (= 0 (find-closest-centroid (trans [1.8421 4.6076]) centroids)))
    (is (= 2 (find-closest-centroid (trans [5.6586 4.8000]) centroids)))
    (is (= 1 (find-closest-centroid (trans [6.3526 3.2909]) centroids)))
    (is (= [0 2 1] (find-closest-centroids (matrix (take 3 X)) centroids)))))

(deftest test-compute-centroids
  (let [approx (approximately 1e-6)
        {:keys [centroids X]} (init-ex7)
        idx (find-closest-centroids X centroids)
        new-centroids (map matrix (compute-centroids X idx 3))]
    (is (approx (matrix [[2.428301 3.157924] [5.813503 2.633656] [7.119387 3.616684]])
          new-centroids))))

(deftest test-run-kmeans
  (let [approx (approximately 1e-4)
        {:keys [centroids X]} (init-ex7)
        centroids (run-kmeans X centroids 10)]
    (is (approx (matrix [[1.9540 5.0256] [3.0437 1.0154] [6.0337 3.0005]]) centroids))))

(deftest test-pca
  (let [approx (approximately 1e-5)
        X (:X (read-dataset-mat5 "data/ex7data1.mat"))
        Xnorm (:data (feature-normalize X))
        U (:U (pca Xnorm))
        Z (project-data Xnorm U 1)
        Xrec (recover-data Z U 1)]
    (is (approx [-0.707107 -0.707107] (sel U :cols 0)))
    (is (approx 1.481274 (first Z)))
    (is (approx [-1.047419 -1.047419] (first Xrec)))))