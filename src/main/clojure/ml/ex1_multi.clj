(ns ml.ex1-multi
  (:use (incanter core io)
        (ml util optim linear)))

(defn init-ex1-multi []
  (let [data (to-matrix (read-dataset "data/ex1data2.txt"))]
    {:X (sel data :except-cols 2) :y (sel data :cols 2)}))

(defn predict-gradient-descent [X y features]
  (let [{x-norm :data mu :mean sigma :sigma} (feature-normalize X)
        theta (gradient-descent (linear-cost-function (add-intercept x-norm) y) [0 0 0] :alpha 1 :max-iter 100)]
    (linear-hypothesis theta (trans (into [1] (div (minus features mu) sigma))))))

(defn predict-normal-equation [X y features]
  (let [theta (normal-equation (add-intercept X) y)]
    (linear-hypothesis theta (trans (into [1] features)))))


