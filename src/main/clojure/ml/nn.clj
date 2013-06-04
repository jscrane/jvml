(ns ml.nn
  (:use (incanter core)
        (ml util logistic)))

(defn predict [Theta1 Theta2 X]
  (let [a (logistic-hypothesis (trans Theta1) (add-intercept X))
        b (logistic-hypothesis (trans Theta2) (add-intercept a))]
    (map max-index b)))

(defn random-matrix [[nrow ncol] epsilon]
  (let [r (fn [_] (- (* 2 epsilon (Math/random)) epsilon))]
    (matrix (partition ncol (take (* nrow ncol) (iterate r (r 0)))))))

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
            reg (* lambda 0.5 (+ (sum (map sum-of-squares Theta1-reg)) (sum (map sum-of-squares Theta2-reg))))]
        {:cost (/ (- reg cost) m)
         :grad [(div (plus delta1 (mult lambda Theta1-reg)) m) (div (plus delta2 (mult lambda Theta2-reg)) m)]}))))

(defn unroll [[r1 c1] [r2 c2] v]
  (let [e1 (* r1 c1)
        a (seq (.toArray v))]
    [(matrix (take e1 a) c1) (matrix (drop e1 a) c2)]))

(defn rollup [mats]
  (.vectorize (matrix (mapcat flatten mats))))
