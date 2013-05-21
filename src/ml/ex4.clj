(ns ml.ex4
  (:use (incanter core)
        (ml util logistic gd matlab)))

(def d (read-dataset-mat5 "data/ex3data1.mat"))
(def W (read-dataset-mat5 "data/ex3weights.mat"))

(defn zero-first-column [mat]
  (bind-columns (zeroes (nrow mat)) (sel mat :except-cols 0)))

(defn sigmoid-gradient [z] (let [s (sigmoid z)] (mult s (minus 1 s))))

(defn neural-net-cost-fn [X Y lambda]
  (let [m (nrow Y) a1 (add-intercept X) m1Y (minus 1 Y)]
    (fn [[Theta1 Theta2]]
      (let [z2 (mmult a1 (trans Theta1))    ; 300ms! (400ms if (mmult Theta1 (trans a1)))
            a2 (add-intercept (sigmoid z2))
            z3 (mmult a2 (trans Theta2))
            a3 (sigmoid z3)
;            cost (sum (map sum (plus (mult Y (log a3)) (mult m1Y (log (minus 1 a3))))))   ; 3000ms
            cost (.zSum (plus (mult Y (log a3)) (mult m1Y (log (minus 1 a3)))))   ; 26ms
            d3 (minus a3 Y)
            d2 (mult (sel (mmult d3 Theta2) :except-cols 0) (sigmoid-gradient z2))
            delta2 (mmult (trans d3) a2)
            delta1 (mmult (trans d2) a1)    ; 900ms!
            Theta1-reg (zero-first-column Theta1)
            Theta2-reg (zero-first-column Theta2)
            reg (* lambda 0.5 (+ (sum (map sum-of-squares Theta1-reg)) (sum (map sum-of-squares Theta2-reg))))
            reg-cost (/ (- reg cost) m)]
        (println reg-cost)
        {:cost reg-cost
         :grad [(div (plus delta1 (mult lambda Theta1-reg)) m) (div (plus delta2 (mult lambda Theta2-reg)) m)]}))))

(def X (:X d))
(def y (map int (:y d)))
(def Y (matrix (map #(boolean-vector 10 %) y)))

(defn predict [theta1 theta2 X]
  (let [a (logistic-hypothesis (trans theta1) (add-intercept X))
        b (logistic-hypothesis (trans theta2) (add-intercept a))]
    (map max-index b)))

(defn random-matrix [[nrow ncol] epsilon]
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
    (let [eps 0.25
          T1 (random-matrix (dim (:Theta1 W)) eps)
          T2 (random-matrix (dim (:Theta2 W)) eps)
          [Theta1 Theta2] (gradient-descent (neural-net-cost-fn X Y 1.0) [T1 T2] :num-iters 50 :alpha 2.25)]
      (println "predict" (double (accuracy (predict Theta1 Theta2 X) y))))))
