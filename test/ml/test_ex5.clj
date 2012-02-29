(ns ml.test-ex5
  (:use (clojure test)
        (ml util test-util ex5)))

(def approx (approximately 1e-5))

(deftest regularized-linear-regression
  (let [{cost :cost grad :grad} ((linear-reg-cost-function X y) 1 [1 1])]
    (is (approx 303.993 cost))
    (is (approx [-15.303 598.25] grad))))

(deftest test-set-error
  (let [{Xp :data mean :mean sigma :sigma} (feature-normalize (polynomial-features (:X d) 8))
        Xpoly (add-intercept Xp)
        Xpoly-val (add-intercept (normalize (polynomial-features (:Xval d) 8) mean sigma))
        [lambdas validation-errors _] (validation-curve Xpoly y Xpoly-val (:yval d))
        lambda-opt (first (apply min-key second (zipmap lambdas validation-errors)))
        theta (train-linear-regression Xpoly y lambda-opt)
        Xpoly-test (add-intercept (normalize (polynomial-features (:Xtest d) 8) mean sigma))]
    (is (= 3 lambda-opt))
    ; this is not the value in the notes because we're not using the same optimization function
    (is (approx 3.8274 (:cost ((linear-reg-cost-function Xpoly-test (:ytest d)) 0 theta))))))