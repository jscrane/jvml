(ns ml.ex1
  (:use (incanter core charts io)
        (ml util optim linear)))

(defn- compute-parameters [X y]
  (gradient-descent (linear-cost-function X y) [0 0] :alpha 0.01 :max-iter 1500))

(defn init-ex1 []
  (let [data (to-matrix (read-dataset "data/ex1data1.txt"))
        y (sel data :cols 1)
        X (add-intercept (sel data :cols 0))]
    { :y y :X X :theta (compute-parameters X y)}))

(defn initial-cost [X y]
  (linear-cost X y [0 0]))

(defn predict-profit [theta v]
  (* (linear-hypothesis theta (trans [1 v])) 10000))

(if *command-line-args*
  (let [{:keys [X y theta]} (init-ex1)]
    (doto (scatter-plot (sel X :cols 1) y :x-label "Population of city in 10,000s" :y-label "Profit in $10,000s"
            :series-label "Training Data" :legend true)
      (add-lines (sel X :cols 1) (mmult X theta) :series-label "Linear Regression")
      (view))))