(ns ml.test-ex2
  (:use (clojure test)
        (ml util test-util)
        [ml.ex2 :only (cost theta predict training-accuracy)]
        [ml.ex2-reg :only (reg-cost reg-accuracy)]))

(deftest ex2
  (is (approx 0.693 (cost [0 0 0])))
  (is (approx 0.2035 (cost theta)))
  (is (approx 0.776 (predict [45 85])))
  (is (approx 0.89 (training-accuracy))))

(deftest ex2-reg
  (is (approx 0.693 (reg-cost (zeroes 28))))
  (is (approx 0.83 (reg-accuracy))))