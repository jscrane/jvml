(ns ml.test-ex6
  (:use (clojure test)
        (ml test-util ex6b)))

(def approx (approximately 1e-5))

(deftest test-gaussian-kernel
  (is (approx 0.324652 (eval-gaussian-kernel [1 2 1] [0 4 -1] 2))))