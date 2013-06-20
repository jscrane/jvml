(ns ml.svm
  (:use (incanter core)
        (ml util))
  (:import (edu.berkeley.compbio.jlibsvm.kernel GaussianRBFKernel)
           (edu.berkeley.compbio.jlibsvm ImmutableSvmParameterGrid)
           (edu.berkeley.compbio.jlibsvm.util SparseVector)
           (edu.berkeley.compbio.jlibsvm.binary C_SVC MutableBinaryClassificationProblemImpl)))

; http://lilyx.net/2011/07/02/using-svm-support-vector-machine-from-clojure/
(defmacro set-all! [obj m]
  `(do ~@(map (fn [e] `(set! (. ~obj ~(key e)) ~(val e))) m) ~obj))

(defn sparse-vector [v]
  (let [n (count v) sv (SparseVector. n)]
    (set! (.indexes sv) (int-array (range n)))
    (set! (.values sv) (float-array v))
    sv))

(defn add-examples [prob X y]
  (doseq [ex (map list X y)]
    (.addExample prob (sparse-vector (first ex)) (second ex)))
  prob)

(defn make-params [epsilon C kernel]
  (let [builder (ImmutableSvmParameterGrid/builder)]
    (set-all! builder {eps epsilon Cset #{C} kernelSet #{kernel}})
    (.build builder)))

(defn gaussian-kernel [sigma]
  (GaussianRBFKernel. (float (/ 1 2 sigma sigma))))

(defn- to-boolean [y]
  (vec (map #(if (zero? (int %)) Boolean/FALSE Boolean/TRUE) y)))

(defn train-model [X y C kernel]
  (let [param (make-params 1.0e-3 (float C) kernel)
        problem (MutableBinaryClassificationProblemImpl. Boolean (count y))]
    (.train (C_SVC.) (add-examples problem (to-list X) (to-boolean y)) param)))

; see svmTrain.m: only pick SVs with alpha > 0
(defn model-vectors [model]
  (let [idx (map first (filter #(pos? (second %)) (map-indexed vector (.alphas model))))
        vects (vec (map #(seq (.values %)) (.SVs model)))]
    (map vects idx)))

(defn svm-predict [model Xval]
  (map #(if (.predictLabel model (sparse-vector %)) 1 0) Xval))

(defn optimal-model [X y Xval yval values]
  (apply max-key :accuracy (for [C values sigma values]
                             (let [model (train-model X y C (gaussian-kernel sigma))
                                   acc (accuracy (svm-predict model Xval) (map int yval))]
                               {:model model :accuracy acc :C C :sigma sigma}))))