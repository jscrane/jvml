; https://gist.github.com/1737472
; (see https://gist.github.com/1724396 for a version using incanter's non-linear-model)
(ns ml1.ex1
  (:use (incanter core charts io)
        (ml util gd linear)))

(def data (to-matrix (read-dataset "src/ml1/ex1data1.txt")))
(def y (sel data :cols 1))
(def X (add-intercept (sel data :cols 0)))

(println "initial cost" (linear-cost X y [0 0]))

(def theta (gradient-descent linear-hypothesis X y [0 0] :alpha 0.01 :num-iters 1500))
(println "optimized cost" (linear-cost X y theta) theta)

(println "predict1" (* (linear-hypothesis theta (trans [1 3.5])) 10000))
(println "predict2" (* (linear-hypothesis theta (trans [1 7])) 10000))

(doto (scatter-plot (sel X :cols 1) y :x-label "Population of city in 10,000s" :y-label "Profit in $10,000s"
        :series-label "Training Data" :legend true)
  (add-lines (sel X :cols 1) (mmult X theta) :series-label "Linear Regression")
  (view))
