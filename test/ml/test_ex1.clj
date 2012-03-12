(ns ml.test-ex1
  (:use (clojure test)
        (ml testutil)
        [ml.ex1 :only (initial-cost predict-profit)]
        [ml.ex1-multi :only (predict-gradient-descent predict-normal-equation)]))

(def approx (approximately 1e-5))

(deftest ex1
  (is (approx 32.073 (initial-cost)))
  (is (approx 4519.8  (predict-profit 3.5)))
  (is (approx 45342  (predict-profit 7))))

(deftest ex1-multi
  (is (approx 293081 (predict-gradient-descent [1650 3])))
  (is (approx 293081 (predict-normal-equation [1650 3]))))
