(ns ml.ex4.ex4
  (:use (incanter core)
        (ml util logistic gd matlab)))

(def d (read-dataset-mat5 "ex4data1.mat"))
(def w (read-dataset-mat5 "ex4weights.mat"))

(defn boolean-vector [n i]
  (into [] (map #(if (= % i) 1 0) (range 1 (inc n)))))

(defn zero-first-column [mat]
  (bind-columns (zeroes (nrow mat)) (sel mat :except-cols 0)))

(defn sigmoid-gradient [z] (let [s (sigmoid z)] (mult s (minus 1 s))))

(defn neural-net-cost-fn [X Y lambda]
  (let [m (nrow Y) a1 (add-intercept X)]
    (fn [[Theta1 Theta2]]
      (let [z2 (mmult a1 (trans Theta1))
            a2 (add-intercept (sigmoid z2))
            z3 (mmult a2 (trans Theta2))
            a3 (sigmoid z3)
            cost (sum (map sum (plus (mult Y (log a3)) (mult (minus 1 Y) (log (minus 1 a3))))))
            d3 (minus a3 Y)
            d2 (mult (sel (mmult d3 Theta2) :except-cols 0) (sigmoid-gradient z2))
            delta2 (mmult (trans d3) a2)
            delta1 (mmult (trans d2) a1)
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

(println "feed-forward" (:cost ((neural-net-cost-fn X Y 0) [(:Theta1 w) (:Theta2 w)])))
(println "regularization" (:cost ((neural-net-cost-fn X Y 1) [(:Theta1 w) (:Theta2 w)])))
(println "sigmoid-gradient" (sigmoid-gradient [1 -0.5 0 0.5 1]))

; checkNNGradients
(defn debug-matrix [nrow ncol]
  (trans (matrix (map #(/ (Math/sin %) 10) (range 1 (inc (* nrow ncol)))) nrow)))

(let [hidden 5 input 3 labels 3 m 5
      T1 (debug-matrix hidden (inc input)) T2 (debug-matrix labels (inc hidden))
      X (debug-matrix m input)
      Y (matrix (map #(boolean-vector labels %) (map #(inc (rem % labels)) (range 1 (inc m)))))]
  (println "costfn (debug)" (:cost ((neural-net-cost-fn X Y 0) [T1 T2])))
  (println "regularized" (:cost ((neural-net-cost-fn X Y 3) [T1 T2]))))

(defn predict [theta1 theta2 X]
  (let [a (logistic-hypothesis (trans theta1) (add-intercept X))
        b (logistic-hypothesis (trans theta2) (add-intercept a))]
    (map max-index (to-vect b))))

(defn random-seq [epsilon]
  (let [r (fn [_] (- (* 2 epsilon (Math/random)) epsilon))]
    (iterate r (r 0))))

(defn random-matrix [[nrow ncol] epsilon]
  (let [rand (random-seq epsilon)]
    (matrix (for [r (range nrow)] (take ncol rand)))))

(time
  (let [eps 0.12
        T1 (random-matrix (dim (:Theta1 w)) eps)
        T2 (random-matrix (dim (:Theta2 w)) eps)
        [Theta1 Theta2] (gradient-descent (neural-net-cost-fn X Y 0.1) [T1 T2] :num-iters 50 :alpha 2.0)]
    (println "theta1" Theta1)
    (println "predict" (double (accuracy (predict Theta1 Theta2 X) y)))))

; iter  %     s
; 5     11   22
; 25   32   117
; 50   50   219
; 100 57   456
; 200 90   871
; 400 94  1711
; 800 97  3437
; 1600 98 6937