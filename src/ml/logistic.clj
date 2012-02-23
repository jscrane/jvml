(ns ml.logistic
  (:gen-class)
  (:use (incanter core)))

(defn sigmoid [^Double z] (/ 1 (+ 1 (exp (- z)))))

(defn logistic-hypothesis [theta X]
  (let [m (mmult X theta)]
    (if (matrix? m) (map sigmoid m) (sigmoid m))))

(defn logistic-cost [X y theta]
  (let [h (logistic-hypothesis theta X) m (nrow y)]
    (/ (reduce - (map #(if (zero? %2) (log (- 1 %1)) (log %1)) h y)) m)))

(defn prediction [^Double v] (map #(if (< % 0.5) 0 1) v))