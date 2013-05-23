(ns ml.kmeans
  (:use (ml util)
        (incanter core stats)))

(defn find-closest-centroid [point centroids]
  (indexes-of? < (map #(sum-of-squares (minus point %)) centroids)))

(defn find-closest-centroids [X centroids]
  (map #(find-closest-centroid % centroids) (to-list X)))

(defn compute-centroid [X idx k]
  (map mean (trans (remove nil? (map #(if (= k %2) %1) (to-list X) idx)))))

(defn compute-centroids [X idx k]
  (matrix (map #(compute-centroid X idx %) (range k))))

(defn kmeans [X centroids]
  (let [k (nrow centroids)
        idx (find-closest-centroids X centroids)]
    (compute-centroids X idx k)))

(defn run-kmeans [X initial-centroids n]
  (first (drop n (iterate #(kmeans X %) initial-centroids))))

(defn init-centroids [X k]
  (sel X :rows (take k (permute (range (nrow X))))))