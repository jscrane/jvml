(ns ml.logistic
  (:gen-class )
  (:import (incanter Matrix))
  (:use (incanter core) (ml util)))

(defn sigmoid [^Matrix z] (div 1 (plus 1 (exp (minus z)))))

(defn logistic-hypothesis [^Matrix theta ^Matrix X] (sigmoid (mmult X theta)))

(defn logistic-cost [^Matrix X ^Matrix y ^Matrix theta]
  (let [h (logistic-hypothesis theta X) m (nrow y) o (ones m)]
    (- (/ (sum (plus (mult (log h) y) (mult (minus o y) (log (minus o h))))) m))))

(defn prediction [v] (map #(if (< % 0.5) 0 1) v))

(defn accuracy [p y] (/ (count (filter true? (map = p y))) (nrow y)))
