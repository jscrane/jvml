(ns kaggle.titanic.random-forest
  (:import (mlclass.randomforest RandomForest))
  (:use [kaggle.titanic.data :only (init submit)]
        [incanter.core :only (bind-columns)]
        [ml.util :only (accuracy)]))

(defn random-forest [n C Xtrain ytrain feature-names]
  (let [vects (map #(.vectorize %) (bind-columns Xtrain ytrain))
        brf (RandomForest. n C (count (first Xtrain)) (apply list vects))]
    {:evaluate (fn [X] (map #(.evaluate brf (.vectorize %)) X))
     :error (.error brf)
     :importances (into (sorted-map) (zipmap feature-names (.importances brf)))}))

(time
  (let [features [:pclass :sex :title]
        {:keys [y X Xtest]} (init 0 features)
        {:keys [evaluate error importances]}
        (reduce (fn [{error :error :as best} _]
                  (let [curr (random-forest 75 2 X y features)]
                    (println _ error)
                    (if (> error (:error curr)) curr best)))
          {:error 1} (range 150))]
    (println "error:" error)
    (println "importances:" importances)
    (println "training:" (double (accuracy (evaluate X) y)))
    (submit (evaluate Xtest))))
