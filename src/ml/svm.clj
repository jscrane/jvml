(ns ml.svm
  (:import (edu.berkeley.compbio.jlibsvm.kernel GaussianRBFKernel)
           (edu.berkeley.compbio.jlibsvm ImmutableSvmParameterGrid)
           (edu.berkeley.compbio.jlibsvm.util SparseVector)))

; http://lilyx.net/2011/07/02/using-svm-support-vector-machine-from-clojure/
(defmacro set-all! [obj m]
  `(do ~@(map (fn [e] `(set! (. ~obj ~(key e)) ~(val e))) m) ~obj))

(defn sparse-vector [v]
  (let [n (count v) sv (SparseVector. n)]
    (do
      (set! (.indexes sv) (int-array (range n)))
      (set! (.values sv) (float-array v))
      sv)))

(defn add-examples [prob X y]
  (doseq [ex (map list X y)]
    (.addExample prob (sparse-vector (first ex)) (second ex)))
  prob)

(defn to-boolean [y]
  (map #(if (zero? (int %)) Boolean/FALSE Boolean/TRUE) y))

(defn make-params [epsilon C kernel]
  (let [builder (ImmutableSvmParameterGrid/builder)]
    (set-all! builder {eps epsilon Cset #{C} kernelSet #{kernel}})
    (.build builder)))

(defn gaussian-kernel [sigma] (GaussianRBFKernel. (float (/ 1 2 sigma sigma))))
