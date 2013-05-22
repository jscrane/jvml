(ns ml.ex7-faces
  (:use (ml pca matlab util)))

(time (let [X (:X (read-dataset-mat5 "data/ex7faces.mat"))
      Xnorm (:data (feature-normalize X))
      U (:U (pca Xnorm))
;      Z (project-data Xnorm U 100)
;      Xrec (recover-data Z U 100)
      ]

  ))
