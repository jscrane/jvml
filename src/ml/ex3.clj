(ns ml.ex3
  (:use (incanter core)
        (ml util logistic gd matlab)))

(def d (read-dataset-mat5 "data/ex3data1.mat"))

(defn one-vs-all [X y num-labels lambda iters]
  (let [m (nrow X) X (add-intercept X) initial-theta (zeroes (ncol X))]
    (reduce (fn [all-theta c]
              (let [samples (into [] (map #(if (= % c) 1 0) y))
                    theta (gradient-descent (reg-cost-fn logistic-hypothesis X samples lambda) initial-theta :num-iters iters)]
                (println c (logistic-cost X samples theta))
                (conj all-theta theta)))
      [] (range 1 (inc num-labels)))))

(defn one-vs-all-accuracy [lambda iters]
  (let [y (map int (d :y ))
        all-theta (matrix (one-vs-all (d :X ) y 10 lambda iters))
        s (mmult (add-intercept (d :X )) (trans all-theta))]
    (double (accuracy (map max-index s) y))))

; iters   accy
;  5       65%
;  10     68%
;  50     75%
;  200   80%
;  500   82%
; 1000  84%