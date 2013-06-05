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
; training accuracy 99.825
; test accuracy 98.9
(if *command-line-args*
  (let [{:keys [X y]} (read-dataset-mat5 "data/spamTrain.mat")
        model (train-model X y 0.1 (LinearKernel.))
        train-accuracy (accuracy (svm-predict model X) (to-boolean y))

        {:keys [Xtest ytest]} (read-dataset-mat5 "data/spamTest.mat")
        test-accuracy (accuracy (svm-predict model Xtest) (to-boolean ytest))]

    (println "training accuracy" (double (* train-accuracy 100)))
    (println "test accuracy" (double (* test-accuracy 100)))))
