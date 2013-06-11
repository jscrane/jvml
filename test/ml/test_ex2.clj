(ns ml.test-ex2
  (:use (clojure test)
        (ml util testutil)
        [ml.ex2 :only (init-ex2 cost predict training-accuracy)]
        [ml.ex2-reg :only (init-ex2-reg reg-cost reg-accuracy)]))

(def approx (approximately 1e-3))

(deftest ex2
  (let [{:keys [X y norm theta]} (init-ex2)]
    (is (approx 0.693 (cost X y [0 0 0])))
    (is (approx 0.2035 (cost X y theta)))
    (is (approx 0.776 (predict theta norm [45 85])))
    (is (approx 0.89 (training-accuracy X y theta)))))

(deftest ex2-reg
  (let [{:keys [X y]} (init-ex2-reg)]
    (is (approx 0.693 (reg-cost X y (zeroes 28))))
    (is (approx 0.83 (reg-accuracy X y 1)))
    (is (approx 0.746 (reg-accuracy X y 10)))
    (is (approx 0.61 (reg-accuracy X y 100)))))