(ns ml.ex1-multi
  (:use (incanter core io)
        (ml util optim linear)))

(defn init-ex1-multi []
  (let [data (to-matrix (read-dataset "data/ex1data2.txt"))]
    {:X (sel data :except-cols 2) :y (sel data :cols 2)}))

(defn predict-gradient-descent [args features]
  (let [{:keys [X y]} args
        {x-norm :data mu :mean sigma :sigma} (feature-normalize X)
        theta (gradient-descent (cost-fn linear-cost linear-hypothesis (add-intercept x-norm) y) [0 0 0] :alpha 1 :max-iter 100)]
    (linear-hypothesis theta (trans (into [1] (div (minus features mu) sigma))))))

(defn predict-normal-equation [args features]
  (let [{:keys [X y]} args
        theta (normal-equation (add-intercept X) y)]
    (linear-hypothesis theta (trans (into [1] features)))))


