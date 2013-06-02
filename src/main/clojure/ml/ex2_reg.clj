(ns ml.ex2-reg
  (:use (incanter core charts io)
        (ml util optim logistic)))

; add polynomial features
(defn- map-features [x1 x2]
  (let [m (nrow x1)]
    (bind-columns (repeat m 1)
      (apply bind-columns (for [i (range 1 7) j (range 0 (inc i))] (mult (pow x1 (- i j)) (pow x2 j)))))))

(defn init-ex2-reg []
  (let [data (to-matrix (read-dataset "data/ex2data2.txt"))]
    {:y (map int (sel data :cols 2))
     :X (map-features (sel data :cols 0) (sel data :cols 1))}))

(defn- optimize [X y lambda]
  (let [initial-theta (zeroes (ncol X))]
    (gradient-descent (reg-logistic-cost-function X y lambda) initial-theta :alpha 0.05 :max-iter 5000)))

(defn reg-cost [args theta]
  (let [{:keys [X y]} args]
    (logistic-cost X y theta)))

(defn reg-accuracy [args lambda]
  (let [{:keys [X y]} args
        theta (optimize X y lambda)]
    (double (accuracy (prediction (logistic-hypothesis theta X)) y))))

(defn- linspace [a b n]
  (let [d (/ (- b a) (dec n))]
    (range a (+ b (/ d 2)) d)))

(def gmax 50)
(def grid (into [] (linspace -1 1.5 gmax)))

(defn- crossing [z row col]
  (let [grid-value (fn [row col] (if (or (= row gmax) (= col gmax)) 0 (z (+ (* row gmax) col))))
        v (grid-value row col)]
    (if (or (> 0 (* v (grid-value (inc row) col)))
          (> 0 (* v (grid-value row (inc col)))))
      [(grid row) (grid col)])))

(if *command-line-args*
  (let [{:keys [X y]} (init-ex2-reg)
        theta (optimize X y 1)
        z (into [] (for [u grid v grid] (mmult (map-features [u] [v]) theta)))
        crossings (remove nil? (for [row (range gmax) col (range gmax)] (crossing z row col)))]
    (doto
      (scatter-plot (sel X :cols 1) (sel X :cols 2) :group-by y :series-label "y = 1" :x-label "Microchip Test 1" :y-label "Microchip Test 2" :legend true)
      ; can't add-lines because incanter has no way to turn off auto-sort on the XYSeries...
      (add-points (map first crossings) (map second crossings) :series-label "Decision Boundary")
      (view))))
