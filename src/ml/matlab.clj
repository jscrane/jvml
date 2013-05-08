(ns ml.matlab
  (:gen-class )
  (:use (incanter core))
  (:import (com.jmatio.io MatFileReader)))

(defn- make-matrix [array2d]
  (let [iv (partial into [])]
    (matrix (map iv (map iv array2d)))))

(defn read-dataset-mat5 [file]
  "Reads a matlab binary file."
  (let [content (.getContent (MatFileReader. file))]
    (reduce (fn [d k] (assoc d (keyword k) (make-matrix (-> content (.get k) .getArray)))) {} (keys content))))


