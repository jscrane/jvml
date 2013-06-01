(ns ml.test-ex5
  (:use (clojure test)
        (ml util testutil)
        [ml.ex5 :only (init-ex5 linear-reg-cost-function polynomial-features validation-curve train-linear-regression)]))

(def approx (approximately 1e-5))

(deftest regularized-linear-regression
  (let [{:keys [X y]} (init-ex5)
        {cost :cost grad :grad} ((linear-reg-cost-function (add-intercept X) y 1) [1 1])]
    (is (approx 303.993 cost))
    (is (approx [-15.303 598.25] grad))))

(deftest test-set-error
  (let [{:keys [lambdas X y Xval yval Xtest ytest]} (init-ex5)
        {Xp :data mean :mean sigma :sigma} (feature-normalize (polynomial-features X 8))
        Xpoly (add-intercept Xp)
        Xpoly-val (add-intercept (normalize (polynomial-features Xval 8) mean sigma))
        [validation-errors _] (validation-curve lambdas Xpoly y Xpoly-val yval)
        lambda-opt (first (apply min-key second (zipmap lambdas validation-errors)))
        theta (train-linear-regression Xpoly y lambda-opt)
        Xpoly-test (add-intercept (normalize (polynomial-features Xtest 8) mean sigma))]
    (is (= 3 lambda-opt))
    (is (approx 3.8599 (:cost ((linear-reg-cost-function Xpoly-test ytest 0) theta))))))