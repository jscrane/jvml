(ns ml.ex6a
  (:import (edu.berkeley.compbio.jlibsvm.kernel LinearKernel))
  (:use (incanter core charts)
        (ml matlab svm)))

(if *command-line-args*
  (let [{:keys [X y]} (read-dataset-mat5 "data/ex6data1.mat")
        C 1
        vectors (model-vectors (train-model X y C (LinearKernel.)))]
    (doto
      (scatter-plot (sel X :cols 0) (sel X :cols 1) :x-label "" :y-label "" :title (str "Linear Kernel, C=" C) :group-by y)
      (add-points (map first vectors) (map second vectors))
      (view))))
