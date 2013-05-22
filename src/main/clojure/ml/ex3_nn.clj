(ns ml.ex3-nn
  (:use (incanter core)
        (ml util logistic matlab)))

(defn init-ex3-nn []
  (let [d (read-dataset-mat5 "data/ex3data1.mat")
        w (read-dataset-mat5 "data/ex3weights.mat")]
    {:X (:X d) :y (map int (:y d))
     :Theta1 (:Theta1 w) :Theta2 (:Theta2 w)}))

(defn- predict [Theta1 Theta2 X]
  (let [a (logistic-hypothesis (trans Theta1) (add-intercept X))
        b (logistic-hypothesis (trans Theta2) (add-intercept a))]
    (map max-index b)))

(defn nn-accuracy [args]
  (let [{:keys [Theta1 Theta2 X y]} args]
    (double (accuracy (predict Theta1 Theta2 X) y))))
