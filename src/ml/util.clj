(ns ml.util
  (:gen-class)
  (:use (incanter core stats)))

(defn add-intercept [X] (bind-columns (repeat (nrow X) 1) X))

(def std (comp sqrt variance))

(defn feature-normalize [X]
  (let [m (nrow X) xt (trans X) mu (map mean xt) sigma (map std xt)]
    { :data (div (minus X (conj-rows (repeat m mu))) (repeat m sigma)) :mean mu :sigma sigma}))
