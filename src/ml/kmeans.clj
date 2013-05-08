(ns ml.kmeans
  (:use (ml util)
        (incanter core stats)))

(defn find-closest-centroid [v centroids]
  (let [dv (map #(minus v %) centroids)
        d (map #(mmult % (trans %)) dv)]
    (indexes-of? < d)))

(defn find-closest-centroids [X centroids]
  (map #(find-closest-centroid % centroids) X))

(defn compute-centroid [X idx k]
  (map mean (trans (remove nil? (map #(if (= k %2) %1) X idx)))))

(defn compute-centroids [X idx k]
  (matrix (map #(compute-centroid X idx %) (range k))))

(defn kmeans [X centroids]
  (let [k (nrow centroids)
        idx (find-closest-centroids X centroids)]
    (compute-centroids X idx k)))

(defn run-kmeans [X initial-centroids n]
  (first (drop n (iterate #(kmeans X %) (matrix initial-centroids)))))

(defn init-centroids [X k]
  (sel X :rows (take k (permute (range (nrow X))))))


