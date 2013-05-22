(ns ml.test-ex2
  (:use (clojure test)
        (ml util testutil)
        [ml.ex2 :only (init-ex2 cost predict training-accuracy)]
        [ml.ex2-reg :only (init-ex2-reg reg-cost reg-accuracy)]))

(def approx (approximately 1e-3))

(deftest ex2
  (let [args (init-ex2)]
    (is (approx 0.693 (cost args [0 0 0])))
    (is (approx 0.2035 (cost args (:theta args))))
    (is (approx 0.776 (predict args [45 85])))
    (is (approx 0.89 (training-accuracy args)))))

(deftest ex2-reg
  (let [args (init-ex2-reg)]
    (is (approx 0.693 (reg-cost args (zeroes 28))))
    (is (approx 0.83 (reg-accuracy args 1)))
    (is (approx 0.746 (reg-accuracy args 10)))
    (is (approx 0.61 (reg-accuracy args 100)))))