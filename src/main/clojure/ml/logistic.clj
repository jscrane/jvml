(ns ml.logistic
  (:gen-class )
  (:import (incanter Matrix))
  (:use (incanter core)
        (ml util optim)))

(defn ^Matrix sigmoid [^Matrix z]
  (div 1 (plus 1 (exp (minus z)))))

(defn ^Matrix logistic-hypothesis [^Matrix theta ^Matrix X]
  "The hypothesis function for logistic regression."
  (sigmoid (mmult X theta)))

(defn logistic-cost [^Matrix X ^Matrix y ^Matrix theta]
  "The logistic cost function"
  (let [h (logistic-hypothesis theta X) m (nrow y) o (ones m)]
    (- (/ (sum (plus (mult (log h) y) (mult (minus o y) (log (minus o h))))) m))))

(defn prediction [v]
  "Returns a vector of 0s or 1s, depending on whether the corresponding element of the input vector is < 0.5."
  (map #(if (< % 0.5) 0 1) v))

(defn accuracy [p y]
  "Returns how accurately the prediction vector (p) reflects the labels (y)"
  (/ (count (filter true? (map = p y))) (nrow y)))

(defn logistic-cost-function [X y]
  (cost-fn logistic-cost logistic-hypothesis X y))

(defn reg-logistic-cost-function [X y lambda]
  (reg-cost-fn logistic-cost logistic-hypothesis X y lambda))