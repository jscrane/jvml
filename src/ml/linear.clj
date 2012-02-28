(ns ml.linear
  (:gen-class)
  (:use (incanter core)))

(defn linear-hypothesis [theta X] (mmult X theta))

(defn linear-cost [X y theta]
  (let [h (linear-hypothesis theta X) d (minus h y) m (nrow y)]
    (/ (mmult (trans d) d) 2 m)))

(defn normal-equation [X y]
  (let [Xt (trans X)]
    (mmult (solve (mmult Xt X)) Xt y)))

