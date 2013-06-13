(ns kaggle.titanic.medians
  (:use [incanter.stats :only (median)]))

;
; computing median values and adding where missing
;
(defn- missing-port [port passengers]
  (map #(if (pos? (:embarked? %)) % (assoc % :embarked port)) passengers))

(defn most-common-port [passengers]
  (let [ports (map :embarked (filter (comp pos? :embarked? ) passengers))]
    (partial missing-port (int (median ports)))))

(defn- compute-medians [passengers f keys]
  (reduce (fn [m k] (assoc m k (f passengers k))) {} keys))

(defn- median-fare [passengers {pclass :pclass embarked :embarked}]
  (let [fares (map :fare (filter #(and (= pclass (:pclass %)) (= embarked (:embarked %))) passengers))]
    (median fares)))

(defn- missing-fare [fares passengers]
  (map #(if (pos? (:fare? %)) % (assoc % :fare (fares (select-keys % [:pclass :embarked ])))) passengers))

(defn median-fares [passengers]
  (let [fares (compute-medians passengers median-fare (for [e [0 1 2] c [1 2 3]] {:pclass c :embarked e}))]
    (partial missing-fare fares)))

(defn- median-age [passengers {title :title pclass :pclass}]
  (let [ages (map :age (filter #(and (= pclass (:pclass %)) (= title (:title %))) passengers))]
    (median ages)))

(defn- missing-age [ages passengers]
  (map #(if (pos? (:age? %)) % (assoc % :age (ages (select-keys % [:title :pclass])))) passengers))

(defn median-ages [passengers]
  (let [ages (compute-medians passengers median-age (for [c [1 2 3] t [0 1 2 3 4]] {:pclass c :title t}))]
    (partial missing-age ages)))
