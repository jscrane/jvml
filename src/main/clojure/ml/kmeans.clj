(ns ml.kmeans
  (:use (ml util)
        (incanter core stats)))

(defn find-closest-centroid [centroids point]
  (indexes-of? < (map #(sum-of-squares (minus point %)) centroids)))

(defn find-closest-centroids [centroids X]
  (map (partial find-closest-centroid centroids) (to-list X)))

(defn- update-sums [[sums counts] [point idx]]
  ; note this doall: otherwise the stack blows when the reduction is realised!
  [(assoc sums idx (doall (plus point (sums idx)))) (assoc counts idx (inc (counts idx)))])

(defn compute-centroids [X idx k]
  (let [ic (zeroes k)
        is (vec (repeat k (zeroes (ncol X))))
        [s c] (reduce update-sums [is ic] (map vector (to-list X) idx))]
    (matrix (map div s c))))

(defn- kmeans [X centroids]
  (compute-centroids X (find-closest-centroids centroids X) (nrow centroids)))

(defn run-kmeans [X initial-centroids n]
  (nth (iterate (partial kmeans X) initial-centroids) n))

(defn init-centroids [X k]
  (sel X :rows (take k (permute (range (nrow X))))))