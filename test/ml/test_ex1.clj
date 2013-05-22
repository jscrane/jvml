(ns ml.test-ex1
  (:use (clojure test)
        (ml testutil)
        [ml.ex1 :only (init-ex1 initial-cost predict-profit)]
        [ml.ex1-multi :only (init-ex1-multi predict-gradient-descent predict-normal-equation)]))

(def approx (approximately 1e-5))

(deftest ex1
  (let [args (init-ex1)]
    (is (approx 32.073 (initial-cost args)))
    (is (approx 4519.8 (predict-profit args 3.5)))
    (is (approx 45342 (predict-profit args 7)))))

(deftest ex1-multi
  (let [args (init-ex1-multi)]
    (is (approx 293081 (predict-gradient-descent args [1650 3])))
    (is (approx 293081 (predict-normal-equation args [1650 3])))))
