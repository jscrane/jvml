(ns ml.gd
  (:gen-class )
  (:import (incanter Matrix))
  (:use (incanter core)))

(defn gradient-descent [cost-fn initial-theta & options]
  (let [opts (when options (apply assoc {} options))
        alpha (or (:alpha opts) 0.01)
        num-iters (or (:num-iters opts) 1000)
        mf (fn [theta] (minus theta (mult alpha (:grad (cost-fn theta)))))
        vf (fn [theta] (map #(minus %1 (mult alpha %2)) theta (:grad (cost-fn theta))))]
    (first (drop num-iters (iterate (if (instance? Matrix initial-theta) mf vf) initial-theta)))))

(defn ^Matrix linear-gradient [hf ^Matrix X ^Matrix y theta]
  (let [m (nrow y) h (hf theta X) d (minus h y) xt (trans X)]
    (div (mmult xt d) m)))

(defn cost-fn
  ([hypothesis-fn ^Matrix X ^Matrix y]
    (fn [theta] {:grad (linear-gradient hypothesis-fn X y theta)}))

  ([hypothesis-fn ^Matrix X ^Matrix y lambda]
    (let [lambda (into [0] (repeat (dec (ncol X)) (/ lambda (nrow y))))]
      (fn [theta] {:grad (plus (linear-gradient hypothesis-fn X y theta) (mult theta lambda))}))))
