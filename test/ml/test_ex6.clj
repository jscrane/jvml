(ns ml.test-ex6
  (:use (clojure test)
        (ml testutil)
        [ml.ex6b :only (eval-gaussian-kernel)]
        [ml.ex6-spam :only (email-features)]))

(def approx (approximately 1e-5))

(deftest test-gaussian-kernel
  (is (approx 0.324652 (eval-gaussian-kernel [1 2 1] [0 4 -1] 2))))

(deftest test-email-features
  (let [features (email-features "data/emailSample1.txt")]
    (is (= 1899 (count features)))
    (is (= 45 (apply + features)))))