(ns ml.test-util
  (:use (incanter core)))

(defn- approx [v a b] (> v (Math/abs (/ (- b a) b))))

(defn approximately [tol]
  (fn [a b]
    (if (matrix? b)
      (every? true? (matrix-map #(approx tol %1 %2) a b))
      (approx tol a b))))