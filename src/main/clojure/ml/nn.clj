(ns ml.nn
  (:use (incanter core)
        (ml util logistic)))

(defn predict [Theta1 Theta2 X]
  (let [a (logistic-hypothesis (trans Theta1) (add-intercept X))
        b (logistic-hypothesis (trans Theta2) (add-intercept a))]
    (map max-index b)))

(defn random-matrix [[nrow ncol] epsilon]
  (let [r (fn [_] (- (* 2 epsilon (Math/random)) epsilon))]
    (matrix (partition ncol (take (* nrow ncol) (iterate r (r 0)))))))

