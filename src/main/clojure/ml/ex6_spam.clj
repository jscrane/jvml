(ns ml.ex6-spam
  (:use (ml util matlab svm)
        (incanter core))
  (:import (org.tartarus.martin Stemmer)
           (edu.berkeley.compbio.jlibsvm.kernel LinearKernel)))

(defn- process-file [name]
  (let [replacements
        [["<[^<>]+>" " "]
         ["[0-9]+" "number"]
         ["(http|https)://[^\\s]*" "httpaddr"]
         ["[^\\s]+@[^\\s]+" "emailaddr"]
         ["[$]+" "dollar"]]]
    (remove empty? (.split
                     (reduce (fn [s [a b]] (.replaceAll s a b)) (.toLowerCase (slurp name)) replacements)
                     "[ @/#.-:&*+=\\[\\]?!(){},'\">_<;%\n]"))))

(defn- porter-stemmer [w]
  (let [s (Stemmer.)]
    (dorun (map #(.add s %) w))
    (.stem s)
    (.toString s)))

(defn email-features [file]
  (let [text-words (into #{} (map porter-stemmer (process-file file)))
        vocab (into {} (map #(let [s (.split % "\\t")] [(second s) (first s)]) (.split (slurp "data/vocab.txt") "\\n")))]
    (apply plus (map #(boolean-vector (count vocab) (Integer/parseInt %)) (remove nil? (map vocab text-words))))))

; this takes too long!
(if *command-line-args*
  (let [train-ds (read-dataset-mat5 "data/spamTrain.mat")
        X (:X train-ds) y (:y train-ds) yb (to-boolean y) n (count yb)
        model (train-model X y 0.1 (LinearKernel.))
        train-predict (map #(.predictLabel model (sparse-vector %)) X)
        train-correct (count (filter true? (map = train-predict yb)))

        test-ds (read-dataset-mat5 "data/spamTest.mat")
        test-predict (map #(.predictLabel model (sparse-vector %)) (:Xtest test-ds))
        test-correct (count (filter true? (map = test-predict (to-boolean (:ytest test-ds)))))]

    (println "training accuracy" (/ (double train-correct) n 0.01))
    (println "test accuracy" (/ (double test-correct) (count (:ytest test-ds)) 0.01))))
