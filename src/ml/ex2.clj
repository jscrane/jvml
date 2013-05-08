(ns ml.ex2
  (:use (incanter core charts io)
        (ml util gd logistic)))

(def data (to-matrix (read-dataset "data/ex2data1.txt")))
(def y (map int (sel data :cols 2)))
(def norm (feature-normalize (sel data :except-cols 2)))
(def X (add-intercept (:data norm)))
(def theta (gradient-descent (cost-fn logistic-hypothesis X y) (zeroes 3) :alpha 0.05 :num-iters 20000))

(defn cost [theta] (logistic-cost X y theta))

(defn predict [scores] (logistic-hypothesis theta (trans (into [1] (normalize (vector scores) (:mean norm) (:sigma norm))))))

(defn training-accuracy [] (double (accuracy (prediction (logistic-hypothesis theta X)) y)))

(defn line-y [theta x]
  (let [[t0 t1 t2] theta]
    (/ (+ (* t1 x) t0) (- t2))))

(let [px [(reduce min (sel X :cols 2)) (reduce max (sel X :cols 2))]]
  (doto
    (scatter-plot (sel X :cols 1) (sel X :cols 2) :group-by y :x-label "Admitted" :y-label "Not Admitted")
    (add-lines px (map (partial line-y theta) px))
    (view)))


