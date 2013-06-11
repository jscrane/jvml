(ns ml.ex3
  (:use (incanter core)
        (ml util logistic optim matlab)))

(defn init-ex3 []
  (let [d (read-dataset-mat5 "data/ex3data1.mat")]
    {:y (map int (:y d)) :X (:X d)}))

(defn one-vs-all [X y num-labels lambda iters]
  (let [m (nrow X) X (add-intercept X) initial-theta (zeroes (ncol X))]
    (apply bind-rows
      (for [c (range 1 (inc num-labels))]
        (let [samples (vec (map #(if (= % c) 1 0) y))]
          (gradient-descent (reg-logistic-cost-function X samples lambda) initial-theta :max-iter iters))))))

(defn one-vs-all-accuracy [X y lambda iters]
  (let [all-theta (matrix (one-vs-all X y 10 lambda iters))
        s (mmult (add-intercept X) (trans all-theta))]
    (double (accuracy (map max-index s) y))))

; iter    accy
; 5       65%
; 10      68%
; 50      75%
; 200     80%
; 500     82%
; 1000    84%