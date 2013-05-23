(ns ml.ex7a
  (:use (ml matlab kmeans)
        (incanter core charts)))

(if *command-line-args*
  (let [X (:X (read-dataset-mat5 "data/ex7data2.mat"))
        centroids (run-kmeans X [[3 3] [6 2] [8 5]] 10)
        idx (find-closest-centroids X centroids)]
    (doto
      (scatter-plot (sel X :cols 0) (sel X :cols 1) :group-by idx)
      (add-points (map first centroids) (map second centroids))
      (view))))
