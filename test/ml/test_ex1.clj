(ns ml.test-ex1
  (:use (clojure test)
        (ml testutil)
        [ml.ex1 :only (init-ex1 initial-cost predict-profit)]
        [ml.ex1-multi :only (init-ex1-multi predict-gradient-descent predict-normal-equation)]))

(def approx (approximately 1e-5))

(deftest ex1
  (let [{:keys [X y theta]} (init-ex1)]
    (is (approx 32.073 (initial-cost X y)))
    (is (approx 4519.8 (predict-profit theta 3.5)))
    (is (approx 45342 (predict-profit theta 7)))))

(deftest ex1-multi
  (let [{:keys [X y]} (init-ex1-multi)]
    (is (approx 293081 (predict-gradient-descent X y [1650 3])))
    (is (approx 293081 (predict-normal-equation X y [1650 3])))))
