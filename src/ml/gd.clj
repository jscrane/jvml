(ns ml.gd
  (:gen-class )
  (:import (incanter Matrix))
  (:use (incanter core)))

(defn gradient-descent [cost-fn initial-theta & options]
  (let [opts (when options (apply assoc {} options))
        alpha (or (:alpha opts) 0.01)]
    (loop [i (or (:num-iters opts) 1000) theta initial-theta]
      (if (zero? i)
        theta
        (recur (dec i) (minus theta (mult alpha (:grad (cost-fn theta)))))))))

(defn- ^Matrix gradients [hf ^Matrix X ^Matrix y theta]
  (let [m (nrow y) h (hf theta X) d (minus h y) xt (trans X)]
    (div (mmult xt d) m)))

(defn cost-fn
  ([hypothesis-fn ^Matrix X ^Matrix y]
    (fn [theta] {:grad (gradients hypothesis-fn X y theta)}))
  ([hypothesis-fn ^Matrix X ^Matrix y lambda]
    (let [lambda (into [0] (repeat (dec (ncol X)) (/ lambda (nrow y))))]
      (fn [theta] {:grad (plus (gradients hypothesis-fn X y theta) (mult theta lambda))}))))
