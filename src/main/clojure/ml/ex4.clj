(ns ml.ex4
  (:use (incanter core)
        (ml util logistic optim matlab)))

(defn init-ex4 []
  (let [d (read-dataset-mat5 "data/ex3data1.mat")
        W (read-dataset-mat5 "data/ex3weights.mat")
        y (map int (:y d))]
    {:X (:X d)
     :y y
     :yb (matrix (map #(boolean-vector 10 %) y))
     :Theta1 (:Theta1 W) :Theta2 (:Theta2 W)}))

(defn- zero-first-column [mat]
  (bind-columns (zeroes (nrow mat)) (sel mat :except-cols 0)))

(defn sigmoid-gradient [z] (let [s (sigmoid z)] (mult s (minus 1 s))))

(defn neural-net-cost-fn [X Y lambda]
  (let [m (nrow Y) a1 (add-intercept X) m1Y (minus 1 Y)]
    (fn [[Theta1 Theta2]]
      (let [z2 (mmult a1 (trans Theta1)) ; 300ms! (400ms if (mmult Theta1 (trans a1)))
            a2 (add-intercept (sigmoid z2))
            z3 (mmult a2 (trans Theta2))
            a3 (sigmoid z3)
            cost (.zSum (plus (mult Y (log a3)) (mult m1Y (log (minus 1 a3))))) ; 26ms
            d3 (minus a3 Y)
            d2 (mult (sel (mmult d3 Theta2) :except-cols 0) (sigmoid-gradient z2))
            delta2 (mmult (trans d3) a2)
            delta1 (mmult (trans d2) a1) ; 900ms!
            Theta1-reg (zero-first-column Theta1)
            Theta2-reg (zero-first-column Theta2)
            reg (* lambda 0.5 (+ (sum (map sum-of-squares Theta1-reg)) (sum (map sum-of-squares Theta2-reg))))
            reg-cost (/ (- reg cost) m)]
        (println reg-cost)
        {:cost reg-cost
         :grad [(div (plus delta1 (mult lambda Theta1-reg)) m) (div (plus delta2 (mult lambda Theta2-reg)) m)]}))))

(defn- predict [theta1 theta2 X]
  (let [a (logistic-hypothesis (trans theta1) (add-intercept X))
        b (logistic-hypothesis (trans theta2) (add-intercept a))]
    (map max-index b)))

(defn- random-matrix [[nrow ncol] epsilon]
  (let [r (fn [_] (- (* 2 epsilon (Math/random)) epsilon))]
    (matrix (partition ncol (take (* nrow ncol) (iterate r (r 0)))))))

; iter  cost  predict time
; 50    1.45  82.7%   67
; 100   0.94  89.5%   127
; 200   0.60  93.4%   262
; 400   0.52  95.1%   502
; 800   0.45  96.3%   1030
(comment
  (time
    (let [{:keys [Theta1 Theta2 X yb y]} (init-ex4)
          eps 0.25
          T1 (random-matrix (dim Theta1) eps)
          T2 (random-matrix (dim Theta2) eps)
          [Th1 Th2] (gradient-descent (neural-net-cost-fn X yb 1.0) [T1 T2] :max-iter 50 :alpha 2.25)]
      (println "predict" (double (accuracy (predict Th1 Th2 X) y))))))

(defn- unroll [d1 d2 v]
  (let [r1 (first d1) c1 (second d1) e1 (* r1 c1)
        r2 (first d2) c2 (second d2)
        a (seq (.toArray v))]
    [(matrix (take e1 a) c1) (matrix (drop e1 a) c2)]))

(defn- rollup [mats]
  (.vectorize (matrix (mapcat flatten mats))))

; iter  cost  predict time
; 50    0.49  96.6%   123
; 100   0.36  98.5%   218
; 200   0.34  99.1%   451
; 400   0.32  99.6%   883
(if *command-line-args*
  (time
    (let [{:keys [Theta1 Theta2 X yb y]} (init-ex4)
          eps 0.25
          d1 (dim Theta1) d2 (dim Theta2)
          [Th1 Th2] (fmincg (neural-net-cost-fn X yb 1.0) [(random-matrix d1 eps) (random-matrix d2 eps)]
                      :max-iter 50 :verbose true :reshape [rollup (partial unroll d1 d2)])]
      (println "predict" (double (accuracy (predict Th1 Th2 X) y))))))