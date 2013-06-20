(ns ml.util
  (:gen-class )
  (:import (incanter Matrix))
  (:use (incanter core stats)))

(defn- repeats [n x] (vec (repeat n x)))

(defn ^Matrix add-intercept [^Matrix X] (bind-columns (repeats (nrow X) 1) X))

(def std (comp sqrt variance))

(defn sumsq [^Matrix M]
  (let [v (vec (vectorize M))] (mmult (trans v) v)))

(defn feature-normalize [^Matrix X]
  (let [m (nrow X) xt (trans X) mu (vec (map mean xt)) sigma (vec (map std xt))]
    {:data (div (minus X (matrix (repeats m mu))) (matrix (repeats m sigma)))
     :mean (trans mu) :sigma (trans sigma)}))

(defn normalize [v mu sigma]
  (let [m (nrow v)]
    (div (minus v (conj-rows (repeats m mu))) (repeats m sigma))))

(defn zeroes [^long n] (repeats n 0))

(defn ones [^long n] (repeats n 1))

(defn indexes-of? [pred? coll]
  (first (reduce #(if (pred? (second %1) (second %2)) %1 %2) (map-indexed vector coll))))

(defn max-index [coll] (inc (indexes-of? > coll)))

(defn boolean-vector [n i]
  (vec (map #(if (= % i) 1 0) (range 1 (inc n)))))

(defn accuracy
  "Returns how accurately the prediction vector (p) reflects the labels (y)"
  [p y]
  (/ (count (filter true? (map = p y))) (nrow y)))
