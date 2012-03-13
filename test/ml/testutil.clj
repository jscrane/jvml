(ns ml.testutil
  (:use (incanter core)))

(defmulti apprx (fn [v a b] (type b)))

(defmethod apprx Double
  ([v a b] (> v (Math/abs (/ (- b a) b)))))

(defmethod apprx incanter.Matrix
  ([v a b] (every? true? (map #(apprx v %1 %2) a b))))

(defmethod apprx clojure.lang.LazySeq
  ([v a b] (apprx v a (matrix b))))

(defn approximately [tol] (fn [a b] (apprx tol a b)))
