(ns ml.gd
  (:gen-class )
  (:import (incanter Matrix))
  (:use (incanter core)))

(defn gradient-descent [cost-fn initial-theta & options]
  "Performs gradient descent to find the model parameters which minimize the given cost function.

  initial-theta: the initial model parameters.
  Options:
    :max-iter (default 1000)
      the number of iterations to run
    :alpha (default 0.01)
      the step-size or learning rate"
  (let [opts (when options (apply assoc {} options))
        alpha (or (:alpha opts) 0.01)
        max-iter (or (:max-iter opts) 1000)
        mf (fn [theta]
             (minus theta (mult alpha (:grad (cost-fn theta)))))
        vf (fn [theta]
             (map #(minus %1 (mult alpha %2)) theta (:grad (cost-fn theta))))]
    (first (drop max-iter (iterate (if (instance? Matrix initial-theta) mf vf) initial-theta)))))

(defn ^Matrix linear-gradient [hf ^Matrix X ^Matrix y theta]
  (let [m (nrow y) h (hf theta X) d (minus h y) xt (trans X)]
    (div (mmult xt d) m)))

(defn cost-fn [hypothesis-fn ^Matrix X ^Matrix y]
  "Constructs a cost function.

  hypothesis-fn: a function of two arguments: the model parameters (theta) and the input features (X)
  X: the input features, an mxn Matrix
  y: the input labels, an mx1 Matrix"
  (fn [theta] {:grad (linear-gradient hypothesis-fn X y theta)}))

(defn reg-cost-fn [hypothesis-fn ^Matrix X ^Matrix y lambda]
  "Constructs a regularised cost function.

  hypothesis-fn: a function of two arguments: the model parameters (theta) and the input features (X)
  X: the input features, an mxn Matrix
  y: the input labels, an mx1 Matrix
  lambda: the regularisation parameter, between 0 and 1"
  (let [lambda (into [0] (repeat (dec (ncol X)) (/ lambda (nrow y))))]
    (fn [theta] {:grad (plus (linear-gradient hypothesis-fn X y theta) (mult theta lambda))})))
