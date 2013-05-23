(ns ml.ex6c
  (:use (incanter core charts)
        (ml matlab svm)))

(defn optimal-model [X y Xval yval]
  (let [values [0.01 0.03 0.1 0.3 1 3 10 30]]
    (apply min-key :error (for [C values sigma values]
                            (let [model (train-model X y C (gaussian-kernel sigma))
                                  predict (map #(.predictLabel model (sparse-vector %)) Xval)
                                  error (count (filter false? (map = predict (to-boolean yval))))]
                              {:model model :error error :C C :sigma sigma})))))

(if *command-line-args*
  (let [{:keys [X y Xval yval]}  (read-dataset-mat5 "data/ex6data3.mat")
        vectors (model-vectors (:model (optimal-model X y Xval yval)))]
    (doto
      (scatter-plot (sel X :cols 0) (sel X :cols 1) :group-by y)
      (add-points (map first vectors) (map second vectors))
      (view))))