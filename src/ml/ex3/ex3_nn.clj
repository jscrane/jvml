(ns ml.ex3.ex3-nn
  (:use (incanter core)
        (ml util logistic matlab)))

(def d (read-dataset-mat5 "ex3data1.mat"))
(def w (read-dataset-mat5 "ex3weights.mat"))

(defn max-index [coll]
  (inc (first (reduce #(if (> (second %1) (second %2)) %1 %2) (map-indexed vector coll)))))

(defn predict [theta1 theta2 X]
  (let [a (logistic-hypothesis (trans theta1) (add-intercept X))
        b (logistic-hypothesis (trans theta2) (add-intercept a))]
    (map max-index (to-vect b))))

(println "accuracy" (double (accuracy (predict (w :Theta1) (w :Theta2) (d :X)) (map int (d :y)))))
