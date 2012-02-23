; (see https://gist.github.com/1724426 for an implementation using linear-model)
(ns ml.ex1.ex1-multi
  (:use (incanter core io) (ml util gd linear)))

(def data (to-matrix (read-dataset "ex1data2.txt")))

(let [{X :data mu :mean sigma :sigma} (feature-normalize (sel data :except-cols 2))
      y (sel data :cols 2)
      theta (gradient-descent linear-hypothesis (add-intercept X) y [0 0 0] :alpha 1 :num-iters 100)
      data [1650 3]]
  (println "gd theta" theta)
  (println "gd predict" (linear-hypothesis theta (trans (into [1] (div (minus data mu) sigma))))))

(let [theta (normal-equation (add-intercept (sel data :except-cols 2)) (sel data :cols 2))]
  (println "ne theta" theta)
  (println "ne predict" (linear-hypothesis theta (trans [1 1650 3]))))