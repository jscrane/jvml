(ns kaggle.titanic.random-forest
  (:import (mlclass.randomforest RandomForest))
  (:use [kaggle.titanic.data :only (read-cleanup submit select-features select-interesting)]
        [incanter.core :only (bind-columns dim matrix sel trans)]
        [ml.util :only (accuracy)]))

(defn random-forest [n C Xtrain ytrain feature-names]
  (let [vects (map #(.vectorize %) (bind-columns Xtrain ytrain))
        brf (RandomForest. n C (count (first Xtrain)) (apply list vects))]
    {:evaluate (fn [X] (map #(.evaluate brf (.vectorize %)) X))
     :evaluate-one (fn [v] (.evaluate brf (.vectorize (matrix v))))
     :error (.error brf)
     :importances (into (sorted-map) (zipmap feature-names (.importances brf)))}))

(defn best-forest [X y features iter]
  (reduce (fn [{error :error :as best} _]
            (let [curr (random-forest 75 2 X y features) cerror (:error curr)]
              (if (> error cerror) (do (println _ cerror) curr) best)))
    {:error 1} (range iter)))

(defn train-forest [y train features title]
  ; this relies on :title being the last key in the sorted-set (yuk)
  (let [X (select-features (conj features :title ) train)
        [_ cols] (dim X)
        rows (vec (filter identity (map-indexed #(if (= (int (last %2)) title) %1) X)))
        title-y (sel (matrix y) :rows rows)
        title-X (sel X :rows rows :except-cols (dec cols))]
    (best-forest title-X title-y features 250)))

(defn train-forests [y train title-features]
  (reduce
    (fn [m [title features]] (assoc m title (train-forest y train features title)))
    {} title-features))

(defn evaluate-all [forests data title-features]
  (vec (map
         (fn [datum]
           (let [title (:title datum)
                 row (select-interesting (title-features title) datum)]
             ((:evaluate-one (forests title)) (trans row))))
         data)))

(time
  (let [[training-set test-set] (read-cleanup)
        y (map :survived training-set)
        title-features {1 [:age :family ],
                        2 [:age :pclass :family :embarked ],
                        3 [:age :pclass :family :embarked ],
                        4 [:age :pclass :family :embarked ],
                        5 [:age :pclass :family :embarked ]}
        forests (train-forests y training-set title-features)
        train-predict (evaluate-all forests training-set title-features)
        test-predict (evaluate-all forests test-set title-features)]
    (println "error:" (reverse (map (fn [[k m]] [k (:error m) (:importances m)]) forests)))
    (println "training accuracy:" (double (accuracy train-predict y)))
    (submit test-predict)))
