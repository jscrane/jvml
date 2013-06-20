(ns ml.ex4
  (:use (incanter core)
        (ml util logistic nn optim matlab)))

(defn init-ex4 []
  (let [d (read-dataset-mat5 "data/ex3data1.mat")
        W (read-dataset-mat5 "data/ex3weights.mat")
        y (map int (:y d))]
    {:X (:X d)
     :y y
     :yb (matrix (map #(boolean-vector 10 %) y))
     :Theta1 (:Theta1 W) :Theta2 (:Theta2 W)}))


(defn- random-matrix [M epsilon]
  (let [[nrow ncol] (dim M)
        r (fn [_] (- (* 2 epsilon (Math/random)) epsilon))]
    (matrix (partition ncol (take (* nrow ncol) (iterate r (r 0)))))))

; iter  cost  predict time
; 50    0.49  96.6%   123
; 100   0.36  98.5%   218
; 200   0.34  99.1%   451
; 400   0.32  99.6%   883
(if *command-line-args*
  (time
    (let [{:keys [Theta1 Theta2 X yb y]} (init-ex4)
          eps 0.25
          [T1 T2] (fmincg
                    (neural-net-cost-fn X yb 1.0)
                    [(random-matrix Theta1 eps) (random-matrix Theta2 eps)] :max-iter 50 :verbose true)]
      (println "predict" (double (accuracy (predict T1 T2 X) y))))))