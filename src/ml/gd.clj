; https://gist.github.com/1737468
(ns ml.gd
  (:gen-class)
  (:use (incanter core)))

(defn- gradients [hf X y theta]
  (let [m (nrow y) h (hf theta X) d (minus h y) xt (trans X)]
    (map #(/ (mmult % d) m) xt)))

(defn gradient-descent [hf X y theta & options]
  (let [opts (when options (apply assoc {} options))
        alpha (or (:alpha opts) 0.01)
        lambda (into [0] (repeat (dec (nrow theta)) (/ (or (:lambda opts) 0) (nrow y))))]
    (loop [i (or (:num-iters opts) 1000) theta theta]
      (if (zero? i)
        theta
        (recur (dec i) (minus theta (mult alpha (plus (gradients hf X y theta) (mult theta lambda)))))))))


