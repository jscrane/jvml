(ns ml.ex7-pca
  (:use (incanter core charts)
        (ml matlab util pca)))

(if *command-line-args*
  (let [X (:X (read-dataset-mat5 "data/ex7data1.mat"))
        {Xnorm :data mu :mean} (feature-normalize X)
        p (pca Xnorm)
        U (:U p)
        p (plus mu (mult 1.5 (first (:S p)) (trans (sel U :cols 0))))
        q (plus mu (mult 1.5 (second (:S p)) (trans (sel U :cols 1))))
        Z (project-data Xnorm U 1)
        Xrec (recover-data Z U 1)]
    (doto
      (scatter-plot (sel X :cols 0) (sel X :cols 1))
      (add-lines [(first mu) (first p)] [(second mu) (second p)])
      (add-lines [(first mu) (first q)] [(second mu) (second q)])
      (view))
    (let [p (scatter-plot (sel Xnorm :cols 0) (sel Xnorm :cols 1))]
      (doto p
        (add-points (sel Xrec :cols 0) (sel Xrec :cols 1))
        (view))
      (doall (map #(add-lines p [(first %1) (first %2)] [(second %1) (second %2)]) Xrec Xnorm)))))