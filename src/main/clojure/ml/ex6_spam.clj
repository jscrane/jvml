(ns ml.ex6-spam
  (:use (ml util matlab svm)
        (incanter core))
  (:import (org.tartarus.martin Stemmer)
           (edu.berkeley.compbio.jlibsvm.kernel LinearKernel)
           (edu.berkeley.compbio.jlibsvm.binary C_SVC MutableBinaryClassificationProblemImpl)))

(defn process-file [name]
  (let [replacements
        [["<[^<>]+>" " "]
         ["[0-9]+" "number"]
         ["(http|https)://[^\\s]*" "httpaddr"]
         ["[^\\s]+@[^\\s]+" "emailaddr"]
         ["[$]+" "dollar"]]]
    (remove empty? (.split
                     (reduce (fn [s [a b]] (.replaceAll s a b)) (.toLowerCase (slurp name)) replacements)
                     "[ @/#.-:&*+=\\[\\]?!(){},'\">_<;%\n]"))))

(defn porter-stemmer [w]
  (let [s (Stemmer.)]
    (do
      (dorun (map #(.add s %) w))
      (.stem s)
      (.toString s))))

(defn email-features [file]
  (let [text-words (into #{} (map porter-stemmer (process-file file)))
        vocab (into {} (map #(let [s (.split % "\\t")] [(second s) (first s)]) (.split (slurp "data/vocab.txt") "\\n")))]
    (apply plus (map #(boolean-vector (count vocab) (Integer/parseInt %)) (remove nil? (map vocab text-words))))))

(if *command-line-args*
  (let [train-ds (read-dataset-mat5 "data/spamTrain.mat")
        X (:X train-ds) y (to-boolean (:y train-ds)) n (count y)
        param (make-params 1.0e-3 (float 0.1) (LinearKernel.))
        problem (MutableBinaryClassificationProblemImpl. Boolean n)
        model (.train (C_SVC.) (add-examples problem X y) param)
        train-predict (map #(.predictLabel model (sparse-vector %)) X)
        train-correct (count (filter true? (map = train-predict y)))

        test-ds (read-dataset-mat5 "data/spamTest.mat")
        test-predict (map #(.predictLabel model (sparse-vector %)) (:Xtest test-ds))
        test-correct (count (filter true? (map = test-predict (to-boolean (:ytest test-ds)))))]

    (println "training accuracy" (/ (double train-correct) n 0.01))
    (println "test accuracy" (/ (double test-correct) (count (:ytest test-ds)) 0.01))))
