(ns ml2.ex2
  (:use (incanter core charts io) (ml util gd logistic)))

(def data (to-matrix (read-dataset "src/ml2/ex2data1.txt")))
(def y (sel data :cols 2))
(def norm (feature-normalize (sel data :except-cols 2)))
(def X (add-intercept (:data norm)))

(def theta (gradient-descent logistic-hypothesis X y [0 0 0] :alpha 0.05 :num-iters 20000))

(println "cost" (logistic-cost X y theta))
(println "predict" (logistic-hypothesis theta (trans (into [1] (div (minus [45 85] (:mean norm)) (:sigma norm))))))
(println "accuracy" (/ (count (filter true? (map = (predict (logistic-hypothesis theta X)) y))) (double (nrow y))))

(def plot (scatter-plot (sel X :cols 1) (sel X :cols 2) :group-by y :x-label "Admitted" :y-label "Not Admitted"))

(defn line-y [theta x]
  (let [[t0 t1 t2] theta]
    (/ (+ (* t1 x) t0) (- t2))))

(def px [(reduce min (sel X :cols 2)) (reduce max (sel X :cols 2))])
(add-lines plot px (map (partial line-y theta) px))
(view plot)
