(ns ml.optim
  (:gen-class )
  (:import (incanter Matrix)
           (mlclass.fmincg Tuple CostFunction Fmincg))
  (:use (incanter core)))

(defn gradient-descent
  "
  Performs gradient descent to find the model parameters which minimize the given cost function.

  initial-theta: the initial model parameters.
  Options:
    :max-iter (default 1000)
      the number of iterations to run
    :alpha (default 0.01)
      the step-size or learning rate
  "
  [cost-fn initial-theta & options]
  (let [opts (when options (apply assoc {} options))
        alpha (or (:alpha opts) 0.01)
        max-iter (or (:max-iter opts) 1000)
        mf (fn [theta]
             (minus theta (mult alpha (:grad (cost-fn theta)))))
        vf (fn [theta]
             (map #(minus %1 (mult alpha %2)) theta (:grad (cost-fn theta))))
        f (if (matrix? initial-theta) mf vf)]
    (first (drop max-iter (iterate f initial-theta)))))

(defn fmincg
  "
  Finds the model parameters which minimize the given cost function using the method of Conjugate-Gradients.

  initial-theta: the initial model parameters.
  Options:
    :max-iter (default 100)
      the number of iterations to run
    :verbose (default false)
      whether to display the cost after every round
    :reshape
      functions to rollup/unroll the parameters to the cost-fn
  "
  [cost-fn initial-theta & options]
  (let [opts (when options (apply assoc {} options))
        verbose (or (:verbose opts) false)
        max-iter (or (:max-iter opts) 100)
        [rollup unroll] (or (:reshape opts) [#(.vectorize (matrix %)) #(matrix (.toArray %))])]
    (unroll
      (Fmincg/minimize
        (proxy [CostFunction] []
          (evaluateCost [theta]
            (let [{:keys [cost grad]} (cost-fn (unroll theta))]
              (Tuple. cost (rollup grad)))))
        (rollup initial-theta) max-iter verbose))))


(defn- unroll2
  "
  Unrolls a vector into two matrices.
  "
  [[r1 c1] [r2 c2] v]
  (let [e1 (* r1 c1)
        a (seq (.toArray v))]
    [(matrix (take e1 a) c1) (matrix (drop e1 a) c2)]))

(defn- rollup-mats
  "
  Rolls-up a collection of matrices into a vector.
  "
  [mats]
  (.vectorize (matrix (mapcat flatten mats))))

(defn reshape
  "
  Makes a pair of rollup/unroll functions for :reshape
  "
  [M N]
  [rollup-mats (partial unroll2 (dim M) (dim N))])

(defn ^Matrix linear-gradient [hf ^Matrix X ^Matrix y theta]
  (let [m (nrow y) h (hf theta X) d (minus h y) xt (trans X)]
    (div (mmult xt d) m)))

(defn cost-fn
  "
  Constructs a cost function for use with an optimizer.

  cost-fn: a function of the model parameters (theta)
  hypothesis-fn: a function of two arguments: the model parameters (theta) and the input features (X)
  X: the input features, an mxn Matrix
  y: the input labels, an mx1 Matrix
  "
  [cost-fn hypothesis-fn ^Matrix X ^Matrix y]
  (let [grad (partial linear-gradient hypothesis-fn X y)
        cost (partial cost-fn X y)]
    (fn [theta]
      {:grad (grad theta)
       :cost (cost theta)})))

(defn reg-cost-fn
  "
  Constructs a regularised cost function for use with an optimizer.

  cost-fn: a function of the model parameters (theta)
  hypothesis-fn: a function of two arguments: the model parameters (theta) and the input features (X)
  X: the input features, an mxn Matrix
  y: the input labels, an mx1 Matrix
  lambda: the regularisation parameter, between 0 and 1
  "
  [cost-fn hypothesis-fn ^Matrix X ^Matrix y lambda]
  (let [m (nrow y) lm (/ lambda m)
        lambdav (into [0] (repeat (dec (ncol X)) lm))
        grad (partial linear-gradient hypothesis-fn X y)
        cost (partial cost-fn X y)]
    (fn [theta]
      {:grad (plus (grad theta) (mult theta lambdav))
       :cost (+ (cost theta) (* lm 0.5 (sum (pow (rest theta) 2))))})))
