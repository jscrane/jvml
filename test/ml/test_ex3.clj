(ns ml.test-ex3
  (:use (clojure test)
        (ml testutil)
        [ml.ex3 :only (init-ex3 one-vs-all-accuracy)]
        [ml.ex3-nn :only (init-ex3-nn nn-accuracy)]))

(def approx (approximately 0.001))

(deftest ex3
  (let [args (init-ex3)]
    (is (approx 0.648 (one-vs-all-accuracy args 0.1 5)))
    (is (approx 0.747 (one-vs-all-accuracy args 0.1 50)))))

(deftest ex3-nn
  (let [args (init-ex3-nn)]
    (is (approx 0.975 (nn-accuracy args)))))