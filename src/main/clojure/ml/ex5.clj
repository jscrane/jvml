(ns ml.ex5
  (:use (incanter core charts)
        (ml util optim matlab linear)))

(defn init-ex5 []
  (assoc (read-dataset-mat5 "data/ex5data1.mat") :lambdas [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]))

(defn train-linear-regression [X y lambda]
  (fmincg (reg-linear-cost-function X y lambda) (matrix (zeroes (ncol X)))))

(defn- learning-curve [Xtrain ytrain Xval yval lambda]
  (let [theta (train-linear-regression Xtrain ytrain lambda)
        training-error (:cost ((linear-cost-function Xtrain ytrain) theta))
        validation-error (:cost ((linear-cost-function Xval yval) theta))]
    [training-error validation-error]))

(defn- learning-curves [X y Xval yval lambda]
  (reduce
    (fn [[training-errors validation-errors] [Xtrain ytrain]]
      (let [[train val] (learning-curve Xtrain ytrain Xval yval lambda)]
        [(conj training-errors train) (conj validation-errors val)]))
    [[] []] (map #(vector (matrix (take % X)) (matrix (take % y))) (range 2 (inc (nrow X))))))

(defn polynomial-features [X p]
  (apply bind-columns (map #(pow X %) (range 1 (inc p)))))

(defn validation-curve [lambdas X y Xval yval]
  (let [val-cf (linear-cost-function Xval yval)
        train-cf (linear-cost-function X y)]
    (reduce
      (fn [[validation-errors training-errors] lambda]
        (let [theta (train-linear-regression X y lambda)]
          [(conj validation-errors (:cost (val-cf theta))) (conj training-errors (:cost (train-cf theta)))]))
      [[] []] lambdas)))

(if *command-line-args*
  (let [{:keys [X y Xval yval lambdas]} (init-ex5)
        Xi (add-intercept X)
        [training validation] (learning-curves Xi y (add-intercept Xval) yval 0)
        ords (range 1 (inc (nrow Xi)))]
    (doto
      (scatter-plot X y :title "Linear Fit" :x-label "Change in water level" :y-label "Water flowing out of the dam")
      (add-lines X (mmult Xi (train-linear-regression Xi y 0)))
      (view))
    (doto
      (xy-plot ords training :title "Linear Regression Learning Curve"
        :x-label "Number of examples" :y-label "Error" :series-label "Training" :legend true)
      (add-lines ords validation :series-label "Cross Validation")
      (view))

    (let [{Xp :data mean :mean sigma :sigma} (feature-normalize (polynomial-features X 8))
          Xpoly (add-intercept Xp)
          Xpoly-val (add-intercept (normalize (polynomial-features Xval 8) mean sigma))
          ords (range 2 (inc (nrow Xpoly)))]
      (doseq [lambda [0 1 100]]
        (let [[training validation] (learning-curves Xpoly y Xpoly-val yval lambda)]
          (doto
            (scatter-plot X y :title (str "Polynomial Fit (\u03bb=" lambda ")")
              :x-label "Change in water level" :y-label "Water flowing out of the dam")
            (add-lines X (mmult Xpoly (train-linear-regression Xpoly y lambda)))
            (view))
          (doto
            (xy-plot ords training :title (str "Polynomial Learning Curve (\u03bb=" lambda ")")
              :x-label "Number of examples" :y-label "Error" :series-label "Training" :legend true)
            (add-lines ords validation :series-label "Cross Validation")
            (view))))
      (let [[validation-errors training-errors] (validation-curve lambdas Xpoly y Xpoly-val yval)]
        (doto
          (xy-plot lambdas training-errors :x-label "\u03bb" :y-label "Error" :series-label "Training" :legend true)
          (add-lines lambdas validation-errors :series-label "Cross Validation")
          (view))))))
