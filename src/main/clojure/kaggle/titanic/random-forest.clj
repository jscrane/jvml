(ns kaggle.titanic.random-forest
  (:import (mlclass.randomforest RandomForest))
  (:use
    [kaggle.titanic.data :only (init submit)]
    [incanter.core :only (bind-columns)]
    [ml.util :only (accuracy)]))

(defn random-forest [n C Xtrain ytrain]
  (let [vects (map #(.vectorize %) (bind-columns Xtrain ytrain))
        brf (RandomForest. n C (count (first Xtrain)) (apply list vects))]
    (fn [X] (map #(.evaluate brf (.vectorize %)) X))))

(let [{:keys [y yval X Xval Xtest]} (init 50 #{:age :age? :sex :pclass :sibsp :parch :fare :fare? :embarked :embarked?})
      brf (random-forest 5000 2 X y)]
  (println "training: " (double (accuracy (brf X) y)))
  (println "validation: " (double (accuracy (brf Xval) yval)))
  (submit (brf Xtest)))
