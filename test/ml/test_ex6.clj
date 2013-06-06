(ns ml.test-ex6
  (:use (clojure test)
        (ml matlab testutil)
        [ml.svm :only (optimal-model)]
        [ml.ex6 :only (eval-gaussian-kernel)]
        [ml.ex6-spam :only (email-features)]))

(def approx (approximately 1e-5))

(deftest test-gaussian-kernel
  (is (approx 0.324652 (eval-gaussian-kernel [1 2 1] [0 4 -1] 2))))

(deftest test-optimal-model
  (let [{:keys [X y Xval yval]} (read-dataset-mat5 "data/ex6data3.mat")
        {:keys [C sigma]} (optimal-model X y Xval yval [0.01 0.03 0.1 0.3 1 3 10 30])]
    (is (= 3 C))
    (is (= 0.1 sigma))))

(deftest test-email-features
  (let [features (email-features "data/emailSample1.txt")]
    (is (= 1899 (count features)))
    (is (= 45 (apply + features)))))