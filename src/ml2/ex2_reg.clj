(ns ml2.ex2-reg
  (:use (incanter core charts io) (ml util gd logistic)))

(def data (to-matrix (read-dataset "src/ml2/ex2data2.txt")))
(def y (sel data :cols 2))
(def m (nrow y))

; add polynomial features
(def X (let [X1 (sel data :cols 0) X2 (sel data :cols 1)]
  (bind-columns (repeat m 1) (apply bind-columns (for [i (range 1 7) j (range 0 (inc i))] (mult (pow X1 (- i j)) (pow X2 j)))))))

(def initial-theta (into [] (repeat (ncol X) 0)))

(println "initial cost" (logistic-cost X y initial-theta))

(def lambda 1)

(def theta (gradient-descent logistic-hypothesis X y initial-theta :alpha 0.05 :num-iters 1000))

(println "accuracy" (/ (count (filter true? (map = (predict (logistic-hypothesis theta X)) y))) (double m)))

; FIXME: plot decision boundary
; FIXME: regulatised gradient descent using lambda
(def plot (scatter-plot (sel X :cols 1) (sel X :cols 2) :group-by y :x-label "Microchip Test 1" :y-label "Microchip Test 2"))
(view plot)
