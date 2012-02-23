(ns ml.ex2.ex2-reg
  (:use (incanter core charts io)
        (ml util gd logistic)))

(def data (to-matrix (read-dataset "ex2data2.txt")))
(def y (map int (sel data :cols 2)))
(def m (nrow y))

; add polynomial features
(defn map-features [x1 x2]
  (let [m (nrow x1)]
    (bind-columns (repeat m 1)
      (apply bind-columns (for [i (range 1 7) j (range 0 (inc i))] (mult (pow x1 (- i j)) (pow x2 j)))))))

(def X (map-features (sel data :cols 0) (sel data :cols 1)))

(def initial-theta (zeroes (ncol X)))
(println "initial cost" (logistic-cost X y initial-theta))

(def theta (gradient-descent logistic-hypothesis X y initial-theta :alpha 0.05 :num-iters 5000 :lambda 1))
(println "accuracy" (double (accuracy (prediction (logistic-hypothesis theta X)) y)))

(defn linspace [a b n]
  (let [d (/ (- b a) (dec n))]
    (range a (+ b (/ d 2)) d)))

(def gmax 50)
(def grid (into [] (linspace -1 1.5 gmax)))

(def z (into [] (for [u grid v grid] (mmult (map-features [u] [v]) theta))))

(defn crossing [z row col]
  (let [grid-value (fn [row col] (if (or (= row gmax) (= col gmax)) 0 (z (+ (* row gmax) col))))
        v (grid-value row col)]
    (if (or (> 0 (* v (grid-value (inc row) col)))
          (> 0 (* v (grid-value row (inc col)))))
      [(grid row) (grid col)])))

(let [crossings (remove nil? (for [row (range gmax) col (range gmax)] (crossing z row col)))]
  (doto
    (scatter-plot (sel X :cols 1) (sel X :cols 2) :group-by y :x-label "Microchip Test 1" :y-label "Microchip Test 2" :legend true)
    (add-points (map first crossings) (map second crossings) :series-label "Decision Boundary")
    (view)))
