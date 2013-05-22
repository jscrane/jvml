(ns ml.pca
  (:use (incanter core stats)))

(defn pca [X]
  (decomp-svd (covariance X)))

(defn project-data [X U k]
  (mmult X (sel U :cols (range k))))

(defn recover-data [Z U k]
  (mmult Z (trans (sel U :cols (range k)))))


