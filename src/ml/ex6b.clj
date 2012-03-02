(ns ml.ex6b
  (:import (edu.berkeley.compbio.jlibsvm.binary C_SVC MutableBinaryClassificationProblemImpl))
  (:use (incanter core charts)
        (ml matlab svm)))

(defn eval-gaussian-kernel [x1 x2 sigma]
  (.evaluate (gaussian-kernel sigma) (sparse-vector x1) (sparse-vector x2)))

(let [ds (read-dataset-mat5 "data/ex6data2.mat")
      X (:X ds) y (:y ds)
      param (make-params 1.0e-3 (float 1) (gaussian-kernel 0.1))
      problem (MutableBinaryClassificationProblemImpl. Boolean (count y))
      model (.train (C_SVC.) (add-examples problem X (to-boolean y)) param)
      vectors (map #(seq (.values %)) (.SVs model))]
  (doto
    (scatter-plot (sel X :cols 0) (sel X :cols 1) :group-by y)
    (add-points (map first vectors) (map second vectors))
    (view)))
