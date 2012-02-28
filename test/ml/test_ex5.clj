(ns ml.test-ex5
  (:use (clojure test)
        (ml test-util ex5)))

(def approx (approximately 1e-5))

(deftest regularized-linear-regression
  (let [{cost :cost grad :grad} (linear-reg-cost-function 1 [1 1])]
    (is (approx 303.993 cost))
    (is (approx [-15.303 598.25] grad))))
