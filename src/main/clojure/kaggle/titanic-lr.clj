(ns kaggle.titanic-lr
  (use [kaggle.titanic :only (init)]
    (incanter core charts)
    (ml util logistic optim)))

(defn- train-logistic-regression [X y lambda]
  (fmincg (reg-logistic-cost-function X y lambda) (zeroes (count (first X))) :max-iter 1500))

(defn- learning-curve [Xtrain ytrain Xval yval lambda]
  (let [theta (train-logistic-regression Xtrain ytrain lambda)
        training-error (:cost ((logistic-cost-function Xtrain ytrain) theta))
        validation-error (:cost ((logistic-cost-function Xval yval) theta))]
    [training-error validation-error]))

(defn- learning-curves [ords X y Xval yval lambda]
  (reduce
    (fn [[training-errors validation-errors] [Xtrain ytrain]]
      (let [[train val] (learning-curve Xtrain ytrain Xval yval lambda)]
        [(conj training-errors train) (conj validation-errors val)]))
    [[] []] (map #(vector (matrix (take % X)) (matrix (take % y))) ords)))

(let [{:keys [y yval X Xval Xtest]} (init 850 #{:sex :pclass :sibsp :parch :fare :fare? :embarked :embarked?})
      Xi (add-intercept X)
      Xval (add-intercept Xval)
      Xtest (add-intercept Xtest)
      ords (range 50 (inc (nrow Xi)) 50)
      lambda 10
      [training validation] (learning-curves ords Xi y Xval yval lambda)
      theta (train-logistic-regression Xi y lambda)
      pred (prediction (logistic-hypothesis theta Xtest))]
  (println "training accuracy: " (double (accuracy (prediction (logistic-hypothesis theta Xi)) y)))
  (println "validation accuracy: " (double (accuracy (prediction (logistic-hypothesis theta Xval)) yval)))
;  (spit "submission.txt" (apply str (map #(str % "\n") pred)))
  (doto
    (xy-plot ords training :title "Logistic Regression Learning Curve"
      :x-label "Number of examples" :y-label "Error" :series-label "Training" :legend true)
    (add-lines ords validation :series-label "Cross Validation")
    (view)))
