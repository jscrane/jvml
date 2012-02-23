; https://gist.github.com/1737468
(ns ml.gd
  (:gen-class)
  (:import (incanter Matrix))
  (:use (incanter core)))

(defn- ^Matrix gradients [hf ^Matrix X ^Matrix y theta]
  (let [m (nrow y) h (hf theta X) d (minus h y) xt (trans X)]
    (div (mmult xt d) m)))

(defn gradient-descent [hf ^Matrix X ^Matrix y theta & options]
  (let [opts (when options (apply assoc {} options))
        alpha (or (:alpha opts) 0.01)
        lambda (into [0] (repeat (dec (nrow theta)) (/ (or (:lambda opts) 0) (nrow y))))]
    (loop [i (or (:num-iters opts) 1000) theta theta]
      (if (zero? i)
        theta
        (recur (dec i) (minus theta (mult alpha (plus (gradients hf X y theta) (mult theta lambda)))))))))


