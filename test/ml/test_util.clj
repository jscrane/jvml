(ns ml.test-util
  (:use (incanter core)))

(defmulti approx (fn [v a b] (type b)))

(defmethod approx Double
  ([v a b] (> v (Math/abs (/ (- b a) b)))))

(defmethod approx incanter.Matrix
  ([v a b] (every? true? (map #(approx v %1 %2) a b))))

(defmethod approx clojure.lang.LazySeq
  ([v a b] (approx v a (matrix b))))

(defn approximately [tol] (fn [a b] (approx tol a b)))
