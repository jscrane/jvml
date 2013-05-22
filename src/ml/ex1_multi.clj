(ns ml.ex1-multi
  (:use (incanter core io)
        (ml util gd linear)))

(def data (to-matrix (read-dataset "data/ex1data2.txt")))
(def X (sel data :except-cols 2))
(def y (sel data :cols 2))

(defn predict-gradient-descent [features]
  (let [{X :data mu :mean sigma :sigma} (feature-normalize X)
        theta (gradient-descent (cost-fn linear-hypothesis (add-intercept X) y) [0 0 0] :alpha 1 :max-iter 100)]
    (linear-hypothesis theta (trans (into [1] (div (minus features mu) sigma))))))

(defn predict-normal-equation [features]
  (let [theta (normal-equation (add-intercept X) y)]
    (linear-hypothesis theta (trans (into [1] features)))))


