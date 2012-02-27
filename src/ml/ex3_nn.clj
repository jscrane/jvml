(ns ml.ex3-nn
  (:use (incanter core)
        (ml util logistic matlab)))

(def d (read-dataset-mat5 "data/ex3data1.mat"))
(def w (read-dataset-mat5 "data/ex3weights.mat"))

(defn predict [theta1 theta2 X]
  (let [a (logistic-hypothesis (trans theta1) (add-intercept X))
        b (logistic-hypothesis (trans theta2) (add-intercept a))]
    (map max-index b)))

(defn nn-accuracy [] (double (accuracy (predict (w :Theta1) (w :Theta2) (d :X)) (map int (d :y)))))
