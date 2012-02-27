(ns ml.ex1
  (:use (incanter core charts io)
        (ml util gd linear)))

(def data (to-matrix (read-dataset "data/ex1data1.txt")))
(def y (sel data :cols 1))
(def X (add-intercept (sel data :cols 0)))

(defn initial-cost [] (linear-cost X y [0 0]))

(def theta (gradient-descent (cost-fn linear-hypothesis X y) [0 0] :alpha 0.01 :num-iters 1500))

(defn predict-profit [v] (* (linear-hypothesis theta (trans [1 v])) 10000))

(doto (scatter-plot (sel X :cols 1) y :x-label "Population of city in 10,000s" :y-label "Profit in $10,000s"
        :series-label "Training Data" :legend true)
  (add-lines (sel X :cols 1) (mmult X theta) :series-label "Linear Regression")
  (view))
