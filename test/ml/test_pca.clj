(ns ml.test-pca
  (:use (clojure test)
        (incanter core)
        (ml testutil matlab util pca)))

(def approx (approximately 1e-5))

(deftest test-pca
  (let [X (:X (read-dataset-mat5 "data/ex7data1.mat"))
        Xnorm (:data (feature-normalize X))
        U (:U (pca Xnorm))
        Z (project-data Xnorm U 1)
        Xrec (recover-data Z U 1)]
    (is (approx [-0.707107 -0.707107] (sel U :cols 0)))
    (is (approx 1.481274 (first Z)))
    (is (approx [-1.047419 -1.047419] (first Xrec)))))