(ns ml.ex8
  (:use (ml matlab)
        (incanter core charts stats)))

(defn init-ex8 [data]
  (read-dataset-mat5 data))

; the exercise wants the "second moment of the mean" (i.e,, divide by N)
; while incanter.stats/variance gives us the first (i.e., divide by N-1)
(defn estimate-gaussian [X]
  (let [mu (map mean (trans X))
        xm2 (sq (matrix (map #(minus % (trans mu)) X)))
        sigma2 (div (map sum (trans xm2)) (first (dim X)))]
    [mu sigma2]))

(defn select-threshold [yval pval]
  (let [pmx (apply max pval)
        pmn (apply min pval)
        iyval (map int yval)
        positives (map #(= 1 %) iyval)
        negatives (map zero? iyval)]
    ; we have to reverse the result here for compatibility with the exercise!
    (apply max-key second (reverse (map (fn [eps]
                                          (let [predict (map #(> eps %) pval)
                                                tp (count (filter true? (map #(and %1 %2) predict positives)))
                                                fp (count (filter true? (map #(and %1 %2) predict negatives)))
                                                fn (count (filter true? (map #(and (not %1) %2) predict positives)))
                                                p (+ tp fp) pn (+ tp fn)
                                                prec (if (zero? p) 0 (/ tp p))
                                                rec (if (zero? pn) 1 (/ tp pn))
                                                p+r (+ prec rec)]
                                            [eps (if (zero? p+r) 0 (/ (* 2 prec rec) p+r))]))
                                     (range pmn pmx (/ (- pmx pmn) 1000)))))))

(defn multivariate-gaussian [X mu sigma2]
  (map #(apply * %) (trans (map #(vec (pdf-normal %3 :mean %1 :sd (sqrt %2))) mu sigma2 (trans X)))))

(defn outliers [X eps p]
  (sel X :rows (filter identity (map-indexed #(if (> eps %2) %1) p))))

(if *command-line-args*
  (let [{:keys [X Xval yval]} (init-ex8 "data/ex8data1.mat")
        [mu sigma2] (estimate-gaussian X)
        p (multivariate-gaussian X mu sigma2)
        pval (multivariate-gaussian Xval mu sigma2)
        [eps f1] (select-threshold yval pval)
        outliers (outliers X eps p)]
    (doto
      (scatter-plot (sel X :cols 0) (sel X :cols 1) :x-label "Latency (ms)" :y-label "Throughput (Mb/s)")
      (add-points (sel outliers :cols 0) (sel outliers :cols 1))
      (view))))