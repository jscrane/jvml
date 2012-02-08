; https://gist.github.com/1737472
; (see https://gist.github.com/1724396 for a version using incanter's non-linear-model)
(ns ml1.ex1
  (:use (incanter core charts io) (ml util gd linear)))

(def data (to-matrix (read-dataset "src/ml1/ex1data1.txt")))

(def y (sel data :cols 1))
(def X (add-intercept (sel data :cols 0)))

(def plot (scatter-plot (sel X :cols 1) y
            :x-label "Population of city in 10,000s" :y-label "Profit in $10,000s"
            :series-label "Training Data" :legend true))

(def initial-theta [0 0])

(println "cost" (linear-cost X y initial-theta))

(def theta (gradient-descent linear-hypothesis X y initial-theta :alpha 0.01 :num-iters 1500))
(println theta)

(add-lines plot (sel X :cols 1) (mmult X theta) :series-label "Linear Regression")

(view plot)

(println "predict1" (* (mmult (trans theta) [1 3.5]) 10000))
(println "predict2" (* (mmult (trans theta) [1 7]) 10000))