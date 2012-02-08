; https://gist.github.com/1737468
(ns ml.gd
  (:gen-class)
  (:use (incanter core)))

(defn gradient [hf X y theta]
  (let [m (nrow y) h (hf theta X) d (minus h y) xt (trans X)]
    (map #(/ (mmult % d) m) xt)))

(defn gradient-descent [hf X y theta & options]
  (let [opts (when options (apply assoc {} options))
        alpha (or (:alpha opts) 0.01)
        num-iters (or (:num-iters opts) 1000)]
  (loop [i 0 theta theta]
;    (println theta)
    (if (= i num-iters)
      theta
      (recur (inc i) (minus theta (mult alpha (gradient hf X y theta))))))))


