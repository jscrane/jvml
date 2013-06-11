(ns ml.test-ex3
  (:use (clojure test)
        (ml testutil)
        [ml.ex3 :only (init-ex3 one-vs-all-accuracy)]
        [ml.ex3-nn :only (init-ex3-nn nn-accuracy)]))

(def approx (approximately 0.001))

(deftest ex3
  (let [{:keys [X y]} (init-ex3)]
    (is (approx 0.648 (one-vs-all-accuracy X y 0.1 5)))
    (is (approx 0.747 (one-vs-all-accuracy X y 0.1 50)))))

(deftest ex3-nn
  (let [{:keys [Theta1 Theta2 X y]} (init-ex3-nn)]
    (is (approx 0.975 (nn-accuracy Theta1 Theta2 X y)))))