(ns ml.ex6c
  (:import (edu.berkeley.compbio.jlibsvm.binary C_SVC MutableBinaryClassificationProblemImpl))
  (:use (incanter core charts)
        (ml matlab svm)))

(defn optimal-model [X y Xval yval]
  (let [values [0.01 0.03 0.1 0.3 1 3 10 30]]
    (apply min-key :error (for [C values sigma values]
                            (let [param (make-params 1.0e-3 (float C) (gaussian-kernel sigma))
                                  problem (MutableBinaryClassificationProblemImpl. Boolean (count y))
                                  model (.train (C_SVC.) (add-examples problem X (to-boolean y)) param)
                                  predict (map #(.predictLabel model (sparse-vector %)) Xval)
                                  error (count (filter false? (map = predict (to-boolean yval))))]
                              {:model model :error error :C C :sigma sigma})))))

(if *command-line-args*
  (let [ds (read-dataset-mat5 "data/ex6data3.mat")
        X (:X ds) y (:y ds)
        model (:model (optimal-model X y (:Xval ds) (:yval ds)))
        vectors (map #(seq (.values %)) (.SVs model))]
    (doto
      (scatter-plot (sel X :cols 0) (sel X :cols 1) :group-by y)
      (add-points (map first vectors) (map second vectors))
      (view))))