(ns ml.ex5
  (:use (incanter core charts)
        (ml util gd matlab linear)))

(def d (read-dataset-mat5 "data/ex5data1.mat"))
(def X (add-intercept (:X d)))
(def y (:y d))

(defn linear-reg-cost-function [X y]
  (let [m (nrow y) n (ncol X)
        grad (partial linear-gradient linear-hypothesis X y)]
    (fn [lambda theta]
      (let [lm (/ lambda m) l (into [0] (repeat (dec n) lm))]
        {:cost (+ (linear-cost X y theta) (* lm 0.5 (sum (pow (rest theta) 2))))
         :grad (plus (grad theta) (mult theta l))}))))

(defn train-linear-regression [X y lambda]
  (let [theta (zeroes (ncol X))
        cf (linear-reg-cost-function X y)]
    (gradient-descent (partial cf lambda) theta :num-iters 2500 :alpha 0.001)))

(doto
  (scatter-plot (:X d) y :x-label "Change in water level" :y-label "Water flowing out of the dam")
  (add-lines (:X d) (mmult X (train-linear-regression X y 0)))
  (view))

(defn learning-curve [Xtrain ytrain Xval yval lambda]
  (let [theta (train-linear-regression Xtrain ytrain lambda)
        training-error (:cost ((linear-reg-cost-function Xtrain ytrain) 0 theta))
        validation-error (:cost ((linear-reg-cost-function Xval yval) 0 theta))]
    [training-error validation-error]))

(defn learning-curves [X y Xval yval lambda]
  (loop [i 2 training-errors [] validation-errors []]
    (if (> i (nrow X))
      [training-errors validation-errors]
      (let [[train val] (learning-curve (matrix (take i X)) (matrix (take i y)) Xval yval lambda)]
        (recur (inc i) (conj training-errors train) (conj validation-errors val))))))

(let [[training validation] (learning-curves X y (add-intercept (:Xval d)) (:yval d) 0)
      ords (range 2 (inc (nrow X)))]
  (doto
    (xy-plot ords training :x-label "Number of examples" :y-label "Error" :legend true :series-label "Training")
    (add-lines ords validation :series-label "Validation")
    (view)))