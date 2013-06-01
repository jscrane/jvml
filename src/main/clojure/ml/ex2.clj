(ns ml.ex2
  (:use (incanter core charts io)
        (ml util gd logistic)))

(defn init-ex2 []
  (let [data (to-matrix (read-dataset "data/ex2data1.txt"))
        norm (feature-normalize (sel data :except-cols 2))
        y (map int (sel data :cols 2))
        X (add-intercept (:data norm))]
    {:y y :norm norm :X X
     :theta (gradient-descent (cost-fn logistic-cost logistic-hypothesis X y) (zeroes 3) :alpha 0.05 :max-iter 20000)}))

(defn cost [args theta]
  (let [{:keys [X y]} args]
    (logistic-cost X y theta)))

(defn predict [args scores]
  (let [{:keys [theta norm]} args]
    (logistic-hypothesis theta (trans (into [1] (normalize (vector scores) (:mean norm) (:sigma norm)))))))

(defn training-accuracy [args]
  (let [{:keys [X y theta]} args]
    (double (accuracy (prediction (logistic-hypothesis theta X)) y))))

(defn- line-y [theta x]
  (let [[t0 t1 t2] theta]
    (/ (+ (* t1 x) t0) (- t2))))

(if *command-line-args*
  (let [{:keys [X y theta]} (init-ex2)
        px [(reduce min (sel X :cols 2)) (reduce max (sel X :cols 2))]]
    (doto
      (scatter-plot (sel X :cols 1) (sel X :cols 2) :group-by y :x-label "Admitted" :y-label "Not Admitted")
      (add-lines px (map (partial line-y theta) px))
      (view))))


