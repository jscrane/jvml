(ns ml.test-ex8
  (:use (clojure test)
        (incanter core stats)
        (ml ex8 ex8-cofi util testutil)))

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

(deftest test-reduced-cofi-cost-function
  (let [{:keys [X Theta R Y]} (init-ex8-cofi)
        num_users 4 num_movies 5 num_features 3
        X (sel X :rows (range num_movies) :cols (range num_features))
        Theta (sel Theta :rows (range num_users) :cols (range num_features))
        Y (sel Y :rows (range num_movies) :cols (range num_users))
        R (sel R :rows (range num_movies) :cols (range num_users))
        cost ((cofi-cost-fn Y R 0) [X Theta])
        reg-cost ((cofi-cost-fn Y R 1.5) [X Theta])
        approx (approximately 1e-4)]
    (is (approx 22.224 (:cost cost)))
    (is (approx [-2.529 -0.5682] (take 2 (vectorize (first (:grad cost))))))
    (is (approx [-10.568 -3.051] (take 2 (vectorize (second (:grad cost))))))

    (is (approx 31.344 (:cost reg-cost)))
    (is (approx [-0.9560 0.60308] (take 2 (vectorize (first (:grad reg-cost))))))
    (is (approx [-10.14 -2.2935] (take 2 (vectorize (second (:grad reg-cost))))))))
