(ns ml.fmincg
  (:gen-class)
  (:import (mlclass Tuple CostFunction Fmincg)))

(defn fmincg [cost-fn initial-theta & options]
  (let [opts (when options (apply assoc {} options))
        verbose (or (:verbose opts) false)
        max-iter (or (:max-iter opts) 1000)]
        (Fmincg/minimize
          (proxy [CostFunction] []
            (evaluateCost [theta]
              (let [cost (cost-fn theta)]
                (Tuple. (:cost cost) (:grad cost)))))
          max-iter verbose)))