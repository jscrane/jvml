(ns ml.logistic
  (:gen-class )
  (:import (incanter Matrix))
  (:use (incanter core)
        (ml util optim)))

(defn ^Matrix sigmoid [^Matrix z]
  (div 1 (plus 1 (exp (minus z)))))

(defn ^Matrix logistic-hypothesis
  "The hypothesis function for logistic regression."
  [^Matrix theta ^Matrix X]
  (sigmoid (mmult X theta)))

(defn logistic-cost
  "The logistic cost function"
  [^Matrix X ^Matrix y ^Matrix theta]
  (let [h (logistic-hypothesis theta X) m (nrow y)]
    (- (/ (sum (plus (mult (log h) y) (mult (minus 1 y) (log (minus 1 h))))) m))))

(defn prediction
  "Returns a vector of 0s or 1s, depending on whether the corresponding element of the input vector is < 0.5."
  [v]
  (map #(if (< % 0.5) 0 1) v))

(defn logistic-cost-function [X y]
  (cost-fn logistic-cost logistic-hypothesis X y))

(defn reg-logistic-cost-function [X y lambda]
  (reg-cost-fn logistic-cost logistic-hypothesis X y lambda))