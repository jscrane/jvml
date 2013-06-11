(ns ml.ex3-nn
  (:use (incanter core)
        (ml util logistic nn matlab)))

(defn init-ex3-nn []
  (let [d (read-dataset-mat5 "data/ex3data1.mat")
        w (read-dataset-mat5 "data/ex3weights.mat")]
    {:X (:X d) :y (map int (:y d))
     :Theta1 (:Theta1 w) :Theta2 (:Theta2 w)}))

(defn nn-accuracy [Theta1 Theta2 X y]
  (double (accuracy (predict Theta1 Theta2 X) y)))
