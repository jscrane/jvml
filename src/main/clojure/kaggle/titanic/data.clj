(ns kaggle.titanic.data
  (:use (incanter core io stats)
        [kaggle.titanic.classifiers :only (cleanup-classifiers)]
        [kaggle.titanic.medians :only (cleanup-missing)]))

(defn- read-csv [file]
  (second (second (read-dataset file :header true))))

;
; read the raw csv data and clean it up
; replace missing fields by medians (using entire dataset)
(defn read-cleanup []
  (let [training-data (cleanup-classifiers (read-csv "src/main/clojure/kaggle/titanic/train.csv"))
        test-data (cleanup-classifiers (read-csv "src/main/clojure/kaggle/titanic/test.csv"))
        cleanup (cleanup-missing (concat training-data test-data))
        training (shuffle (map cleanup training-data))
        test (map cleanup test-data)]
    [training test]))

(defn select-interesting [keys row]
  (vec (vals (into (sorted-map) (select-keys row keys)))))

(defn select-features [keys X]
  (matrix (map (partial select-interesting keys) X)))

; shuffle the training set and partition into training and validation sets
(defn init [m-val interesting-keys]
  (let [[training test] (read-cleanup)
        train-y (map :survived training)
        train-X (select-features interesting-keys training)
        test-X (select-features interesting-keys test)]
    {:training training :test test
     :train-y train-y :train-X train-X
     :y (vec (drop m-val train-y)) :yval (vec (take m-val train-y))
     :X (matrix (drop m-val train-X)) :Xval (matrix (take m-val train-X))
     :Xtest (matrix test-X)}))

(defn submit [predictions]
  (spit "submission.txt" (apply str (map #(str % "\n") predictions))))