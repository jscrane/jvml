(ns ml.test-ex4
  (:use (clojure test)
        (ml util test-util)
        [incanter.core :only (trans matrix)]
        [ml.ex4 :only (W X Y neural-net-cost-fn sigmoid-gradient)]))

(deftest cost-function
  (is (approx 0.287629 (:cost ((neural-net-cost-fn X Y 0) [(:Theta1 W) (:Theta2 W)]))))
  (is (approx 0.383770 (:cost ((neural-net-cost-fn X Y 1) [(:Theta1 W) (:Theta2 W)]))))
  (is (approx 0.25 (sigmoid-gradient 0))))

(defn- debug-matrix [nrow ncol]
  (trans (matrix (map #(/ (Math/sin %) 10) (range 1 (inc (* nrow ncol)))) nrow)))

(deftest check-nn-gradients
  (let [hidden 5 input 3 labels 3 m 5
        T1 (debug-matrix hidden (inc input)) T2 (debug-matrix labels (inc hidden))
        X (debug-matrix m input)
        Y (matrix (map #(boolean-vector labels %) (map #(inc (rem % labels)) (range 1 (inc m)))))]
    (is (approx 2.10095 (:cost ((neural-net-cost-fn X Y 0) [T1 T2]))))
    (is (approx 2.14635 (:cost ((neural-net-cost-fn X Y 3) [T1 T2]))))))

