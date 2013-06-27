(ns kaggle.titanic.medians
  (:use [incanter.stats :only (median)]))

;
; computing median values and adding where missing
;
(defn- most-common-port [passengers]
  (let [port (median (map :embarked (filter (comp pos? :embarked? ) passengers)))]
    (fn [pass]
      (if (pos? (:embarked? pass))
        pass
        (assoc pass :embarked port)))))

(defn- compute-medians [passengers f keys]
  (reduce (fn [m k] (assoc m k (f passengers k))) {} keys))

(defn- median-fare [passengers {pclass :pclass embarked :embarked}]
  (median (map :fare (filter #(and (= pclass (:pclass %)) (= embarked (:embarked %))) passengers))))

(defn- median-fares [passengers]
  (let [fares (compute-medians passengers median-fare (for [e [0 1 2] c [1 2 3]] {:pclass c :embarked e}))]
    (fn [pass]
      (if (pos? (:fare? pass))
        pass
        (assoc pass :fare (fares (select-keys pass [:pclass :embarked ])))))))

(defn- median-age [passengers {title :title pclass :pclass}]
  (median (map :age (filter #(and (= pclass (:pclass %)) (= title (:title %))) passengers))))

(defn- median-ages [passengers]
  (let [ages (compute-medians passengers median-age (for [c [1 2 3] t [1 2 3 4 5]] {:pclass c :title t}))]
    (fn [pass]
      (if (pos? (:age? pass))
        pass
        (assoc pass :age (ages (select-keys pass [:title :pclass ])))))))

(defn- name-counts [passengers]
  (let [counts (reduce #(assoc %1 %2 (inc (get %1 %2 0))) {} (map #(select-keys % [:last :first ]) passengers))]
    (fn [pass]
      (let [name-tuple (select-keys pass [:last :first ])
            {:keys [sibsp parch title]} pass
            with-spouse (if (or (= title 1) (= title 2)) (dec (counts name-tuple)) 0)
            siblings (- sibsp with-spouse)]
        (merge {:with-spouse with-spouse :siblings siblings} pass)))))

(defn cleanup-missing [passengers]
  (apply comp (map #(% passengers) [name-counts median-ages median-fares most-common-port])))
