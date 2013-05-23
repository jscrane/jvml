(ns ml.ex6a
  (:import (edu.berkeley.compbio.jlibsvm.kernel LinearKernel))
  (:use (incanter core charts)
        (ml matlab svm)))

(if *command-line-args*
  (let [{:keys [X y]} (read-dataset-mat5 "data/ex6data1.mat")
        vectors (model-vectors (train-model X y 1 (LinearKernel.)))]
    (doto
      (scatter-plot (sel X :cols 0) (sel X :cols 1) :group-by y)
      (add-points (map first vectors) (map second vectors))
      (view))))
