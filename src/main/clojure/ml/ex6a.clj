(ns ml.ex6a
  (:import (edu.berkeley.compbio.jlibsvm.kernel LinearKernel)
           (edu.berkeley.compbio.jlibsvm.binary C_SVC MutableBinaryClassificationProblemImpl))
  (:use (incanter core charts)
        (ml matlab svm)))

(if *command-line-args*
  (let [ds1 (read-dataset-mat5 "data/ex6data1.mat")
        X (:X ds1) y (:y ds1)
        param (make-params 1.0e-3 (float 1) (LinearKernel.))
        problem (MutableBinaryClassificationProblemImpl. Boolean (count y))
        model (.train (C_SVC.) (add-examples problem X (to-boolean y)) param)
        vectors (map #(seq (.values %)) (.SVs model))]
    (doto
      (scatter-plot (sel X :cols 0) (sel X :cols 1) :group-by y)
      (add-points (map first vectors) (map second vectors))
      (view))))
