(ns kaggle.titanic.data
  (:use (incanter core io stats)
        [kaggle.titanic.classifiers :only (cleanup-classifiers)]
        [kaggle.titanic.medians :only (most-common-port median-fares median-ages)]))

(defn- read-csv [file]
  (second (second (read-dataset file :header true))))

;
; read the raw csv data and clean it up
; replace missing fields by medians (using entire dataset)
; shuffle the training set and partition into training and validation sets
(defn init [m-val interesting-keys]
  (let [training-data (cleanup-classifiers (read-csv "src/main/clojure/kaggle/titanic/train.csv"))
        test-data (cleanup-classifiers (read-csv "src/main/clojure/kaggle/titanic/test.csv"))

        all-data (concat training-data test-data)
        port (most-common-port all-data)
        fare (median-fares all-data)
        age (median-ages all-data)

        training (shuffle (-> training-data port fare age))
        test (-> test-data port fare age)
        select-interesting (fn [X] (vec (vals (into (sorted-map) (select-keys X interesting-keys)))))

        train-y (map :survived training)
        train-X (map select-interesting training)
        test-X (map select-interesting test)]
    {:training training :test test
     :train-y train-y :train-X train-X
     :y (vec (drop m-val train-y)) :yval (vec (take m-val train-y))
     :X (matrix (drop m-val train-X)) :Xval (matrix (take m-val train-X))
     :Xtest (matrix test-X)}))

(defn submit [predictions]
  (spit "submission.txt" (apply str (map #(str % "\n") predictions))))