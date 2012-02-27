(ns ml.test-util)

(defn approx
  ([a b] (approx a b 0.001))
  ([a b v] (> v (/ (- b a) b))))
