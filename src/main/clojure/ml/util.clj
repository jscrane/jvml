(ns ml.util
  (:gen-class )
  (:import (incanter Matrix))
  (:use (incanter core stats)))

(defn ^Matrix add-intercept [^Matrix X] (bind-columns (repeat (nrow X) 1) X))

(def std (comp sqrt variance))

(defn feature-normalize [^Matrix X]
  (let [m (nrow X) xt (trans X) mu (map mean xt) sigma (map std xt)]
    {:data (div (minus X (conj-rows (repeat m mu))) (repeat m sigma)) :mean (trans mu) :sigma (trans sigma)}))

(defn normalize [v mu sigma]
  (let [m (nrow v)]
    (div (minus v (conj-rows (repeat m mu))) (repeat m sigma))))

(defn- repeats [n x] (into [] (repeat n x)))

(defn zeroes [^long n] (repeats n 0))

(defn ones [^long n] (repeats n 1))

(defn indexes-of? [pred? coll]
  (first (reduce #(if (pred? (second %1) (second %2)) %1 %2) (map-indexed vector coll))))

(defn max-index [coll] (inc (indexes-of? > coll)))

(defn boolean-vector [n i]
  (into [] (map #(if (= % i) 1 0) (range 1 (inc n)))))
