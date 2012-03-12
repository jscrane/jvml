(ns ml.test-ex3
  (:use (clojure test)
        (ml testutil)
        [ml.ex3 :only (one-vs-all-accuracy)]
        [ml.ex3-nn :only (nn-accuracy)]))

(def approx (approximately 0.001))

(deftest ex3
  (is (approx 0.648 (one-vs-all-accuracy 0.1 5)))
  (is (approx 0.747 (one-vs-all-accuracy 0.1 50))))

(deftest ex3-nn
  (is (approx 0.975 (nn-accuracy))))