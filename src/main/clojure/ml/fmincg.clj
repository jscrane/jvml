(ns ml.fmincg
  (:gen-class )
  (:import (mlclass Tuple CostFunction Fmincg)))

(defn fmincg [cost-fn initial-theta & options]
  (let [opts (when options (apply assoc {} options))
        verbose (or (:verbose opts) false)
        max-iter (or (:max-iter opts) 100)
        [rollup unroll] (or (:reshape opts) [identity identity])]
    (unroll
      (Fmincg/minimize
        (proxy [CostFunction] []
          (evaluateCost [theta]
            (let [{:keys [cost grad]} (cost-fn (unroll theta))]
              (Tuple. cost (rollup grad)))))
        (rollup initial-theta) max-iter verbose))))