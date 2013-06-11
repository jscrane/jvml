(ns ml.ex7-pca
  (:use (incanter core charts)
        (ml matlab util pca)))

(def args (into #{} *command-line-args*))

(if (contains? args "pca")
  (let [X (:X (read-dataset-mat5 "data/ex7data1.mat"))
        {Xnorm :data mu :mean} (feature-normalize X)
        {:keys [U S]} (pca Xnorm)
        p (plus mu (mult 1.5 (first S) (trans (sel U :cols 0))))
        q (plus mu (mult 1.5 (second S) (trans (sel U :cols 1))))
        Z (project-data Xnorm U 1)]
    (doto
      (scatter-plot (sel X :cols 0) (sel X :cols 1) :x-label "" :y-label "" :title "Computed Eigenvectors")
      (add-lines [(first mu) (first p)] [(second mu) (second p)])
      (add-lines [(first mu) (first q)] [(second mu) (second q)])
      (view))
    (let [Xrec (recover-data Z U 1)
          p (scatter-plot (sel Xnorm :cols 0) (sel Xnorm :cols 1) :x-label "" :y-label "" :title "Projected and Recovered after PCA")]
      (add-points p (sel Xrec :cols 0) (sel Xrec :cols 1))
      (doall (map (fn [[xr yr] [xn yn]] (add-lines p [xr xn] [yr yn])) Xrec Xnorm))
      (view p))))

; TODO: computing the covariance matrix takes forever...
(if (contains? args "faces")
  (let [X (:X (read-dataset-mat5 "data/ex7faces.mat"))
        Xnorm (:data (feature-normalize X))   ; 95s
        U (:U (pca Xnorm))
        ;      Z (project-data Xnorm U 100)
        ;      Xrec (recover-data Z U 100)
        ]

    ))