(ns ml.test-ex4
  (:use (clojure test)
        (ml util testutil)
        [incanter.core :only (trans matrix)]
        [ml.nn :only (neural-net-cost-fn sigmoid-gradient)]
        [ml.ex4 :only (init-ex4)]))

(def approx (approximately 1e-6))

(deftest cost-function
  (let [{:keys [X yb Theta1 Theta2]} (init-ex4)]
    (is (approx 0.287629 (:cost ((neural-net-cost-fn X yb 0) [Theta1 Theta2]))))
    (is (approx 0.383770 (:cost ((neural-net-cost-fn X yb 1) [Theta1 Theta2]))))
    (is (approx 0.25 (sigmoid-gradient 0)))))

(defn- debug-matrix [nrow ncol]
  (trans (matrix (map #(/ (Math/sin %) 10) (range 1 (inc (* nrow ncol)))) nrow)))

(deftest check-nn-gradients
  (let [hidden 5 input 3 labels 3 m 5
        T1 (debug-matrix hidden (inc input)) T2 (debug-matrix labels (inc hidden))
        X (debug-matrix m input)
        yb (matrix (map #(boolean-vector labels %) (map #(inc (rem % labels)) (range 1 (inc m)))))]
    (is (approx 2.10095 (:cost ((neural-net-cost-fn X yb 0) [T1 T2]))))
    (is (approx 2.14636 (:cost ((neural-net-cost-fn X yb 3) [T1 T2]))))))

