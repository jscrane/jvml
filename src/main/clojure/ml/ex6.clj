(ns ml.ex6
  (:import (edu.berkeley.compbio.jlibsvm.kernel LinearKernel))
  (:use (incanter core charts)
        (ml util matlab svm)))

(defn eval-gaussian-kernel [x1 x2 sigma]
  (.evaluate (gaussian-kernel sigma) (sparse-vector x1) (sparse-vector x2)))

(defn optimal-model [X y Xval yval]
  (let [values [0.01 0.03 0.1 0.3 1 3 10 30]]
    (apply max-key :accuracy (for [C values sigma values]
                               (let [model (train-model X y C (gaussian-kernel sigma))
                                     acc (accuracy (svm-predict model Xval) (to-boolean yval))]
                                 {:model model :accuracy acc :C C :sigma sigma})))))

(if *command-line-args*
  (do
    (let [{:keys [X y]} (read-dataset-mat5 "data/ex6data1.mat")
          C 1
          vectors (model-vectors (train-model X y C (LinearKernel.)))]
      (doto
        (scatter-plot (sel X :cols 0) (sel X :cols 1) :x-label "" :y-label "" :title (str "Linear Kernel C=" C) :group-by y)
        (add-points (map first vectors) (map second vectors))
        (view)))
    (let [{:keys [X y]} (read-dataset-mat5 "data/ex6data2.mat")
          C 1 sigma 0.1
          vectors (model-vectors (train-model X y C (gaussian-kernel sigma)))]
      (doto
        (scatter-plot (sel X :cols 0) (sel X :cols 1)
          :x-label "" :y-label "" :title (str "Gaussian Kernel \u03c3=" sigma " C=" C) :group-by y)
        (add-points (map first vectors) (map second vectors))
        (view)))
    (let [{:keys [X y Xval yval]} (read-dataset-mat5 "data/ex6data3.mat")
          opt (optimal-model X y Xval yval)
          vectors (model-vectors (:model opt))]
      (doto
        (scatter-plot (sel X :cols 0) (sel X :cols 1)
          :x-label "" :y-label "" :title (str "Gaussian Kernel \u03c3=" (:sigma opt) " C=" (:C opt)) :group-by y)
        (add-points (map first vectors) (map second vectors))
        (view)))))
