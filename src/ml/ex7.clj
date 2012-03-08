(ns ml.ex7
  (:use (ml util matlab)
        (incanter core stats charts)))

(def ds (read-dataset-mat5 "data/ex7data2.mat"))

(defn find-closest-centroid [v centroids]
  (let [dv (map #(minus v %) centroids) d (map #(mmult % (trans %)) dv)]
    (indexes-of? < d)))

(defn find-closest-centroids [X centroids]
  (map #(find-closest-centroid % centroids) X))

(defn compute-centroid [X idx k]
  (map mean (trans (remove nil? (map #(if (= k %2) %1) X idx)))))

(defn compute-centroids [X idx k]
  (matrix (map #(compute-centroid X idx %) (range 1 (inc k)))))

(defn kmeans [X centroids]
  (let [k (nrow centroids)
        idx (find-closest-centroids X centroids)]
    (compute-centroids X idx k)))

(defn run-kmeans [X initial-centroids max-iters]
  (loop [i max-iters centroids (matrix initial-centroids)]
    (if (zero? i)
      centroids
      (recur (dec i) (kmeans X centroids)))))

(let [X (:X ds)
      centroids (run-kmeans X [[3 3] [6 2] [8 5]] 10)
      idx (find-closest-centroids X centroids)]
  (doto
    (scatter-plot (sel X :cols 0) (sel X :cols 1) :group-by idx)
    (add-points (map first centroids) (map second centroids))
    (view)))
