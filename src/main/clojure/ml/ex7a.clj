(ns ml.ex7a
  (:use (ml matlab kmeans)
        (incanter core charts)))

(defn init-ex7 []
  (assoc (read-dataset-mat5 "data/ex7data2.mat")
    :centroids (matrix [[3 3] [6 2] [8 5]])))

(if *command-line-args*
  (let [{:keys [centroids X]} (init-ex7)
        idx (find-closest-centroids X (run-kmeans X centroids 10))]
    (doto
      (scatter-plot (sel X :cols 0) (sel X :cols 1) :x-label "" :y-label "" :group-by idx)
      (add-points (map first centroids) (map second centroids))
      (view))))
