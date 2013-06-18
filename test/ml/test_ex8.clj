(ns ml.test-ex8
  (:use (clojure test)
        (incanter core)
        (ml ex8 testutil)))

(deftest test-estimate-gaussian
  (let [{:keys [X Xval yval]} (init-ex8 "data/ex8data1.mat")
        [mu sigma2] (estimate-gaussian X)
        approx (approximately 1e-4)]
    (is (approx [14.112 14.998] mu))
    (is (approx [1.8326 1.7097] sigma2))))

(deftest test-select-threshold-outliers
  (let [{:keys [X Xval yval]} (init-ex8 "data/ex8data1.mat")
        [mu sigma2] (estimate-gaussian X)
        pval (multivariate-gaussian Xval mu sigma2)
        [epsilon f1] (select-threshold yval pval)
        p (multivariate-gaussian X mu sigma2)
        outliers (outliers X epsilon p)
        approx (approximately 1e-4)]
    (is (approx 8.99e-5 epsilon))
    (is (= (/ 7 8) f1))
    (is (= 6 (count outliers)))))

(deftest test-select-threshold-outliers2
  (let [{:keys [X Xval yval]} (init-ex8 "data/ex8data2.mat")
        [mu sigma2] (estimate-gaussian X)
        pval (multivariate-gaussian Xval mu sigma2)
        [epsilon f1] (select-threshold yval pval)
        p (multivariate-gaussian X mu sigma2)
        outliers (outliers X epsilon p)
        approx (approximately 1e-4)]
    (is (approx 1.3772e-18 epsilon))
    (is (approx 0.6154 (double f1)))
    (is (= 117 (count outliers)))))
