(ns ml.test-ex3
  (:use (clojure test)
        (ml test-util)
        [ml.ex3 :only (one-vs-all-accuracy)]
        [ml.ex3-nn :only (nn-accuracy)]))

(deftest ex3
  (is (approx 0.65 (one-vs-all-accuracy 0.1 5)))
  (is (approx 0.75 (one-vs-all-accuracy 0.1 50))))

(deftest ex3-nn
  (is (approx 0.975 (nn-accuracy))))