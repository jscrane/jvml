(ns ml.ex7b
  (:import (javax.swing JFrame JSplitPane JLabel ImageIcon)
           (java.awt.image BufferedImage))
  (:use (ml matlab kmeans)
        (incanter core charts)))

(defn rgb [r g b] (int (+ (* r 65536) (* g 256) b)))

(defn make-image [rows cols coll]
  (let [img (BufferedImage. cols rows BufferedImage/TYPE_INT_RGB)
        data (into [] coll)]
    (doall
      (for [i (range (count data))]
        (let [r (int (/ i rows)) c (rem i rows)]
          (.setRGB img c r (data i)))))
    img))

(let [A (:A (read-dataset-mat5 "data/bird_small.mat"))
      rows (nrow A) cols (/ (ncol A) 3)
      r (vectorize (trans (sel A :cols (range 0 cols))))
      g (vectorize (trans (sel A :cols (range cols (* cols 2)))))
      b (vectorize (trans (sel A :cols (range (* cols 2) (* cols 3)))))

      X (div (bind-columns r g b) 255)
      k 16
      centroids (run-kmeans X (init-centroids X k) 10)
      idx (find-closest-centroids X centroids)
      rgb-centroids (map (fn [[r g b]] (rgb (int (* 255 r)) (int (* 255 g)) (int (* 255 b)))) centroids)]
  (doto (JFrame.)
    (.add
      (JSplitPane. JSplitPane/HORIZONTAL_SPLIT
        (JLabel. (ImageIcon. (make-image rows cols (map rgb r g b))))
        (JLabel. (ImageIcon. (make-image rows cols (map #(nth rgb-centroids %) idx))))))
    (.setDefaultCloseOperation JFrame/EXIT_ON_CLOSE)
    (.pack)
    (.show)))

