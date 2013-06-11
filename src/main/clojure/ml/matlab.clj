(ns ml.matlab
  (:gen-class )
  (:use (incanter core))
  (:import (com.jmatio.io MatFileReader)))

(defn read-dataset-mat5
  "Reads a matlab binary file."
  [file]
  (let [content (.getContent (MatFileReader. file))]
    (reduce
      (fn [d k]
        (assoc d (keyword k) (matrix (vec (map vec (-> content (.get k) .getArray))))))
      {} (keys content))))
