(ns ml.ex8-cofi
  (:use [clojure.java.io :only (reader)]
        (ml util matlab optim)
        (incanter core stats)))

(defn init-ex8-cofi []
  (conj (read-dataset-mat5 "data/ex8_movies.mat")
    (read-dataset-mat5 "data/ex8_movieParams.mat")))

(defn cofi-cost-fn [Y R lambda]
  (fn [[X Theta]]
    (let [D (minus (mmult X (trans Theta)) Y)
          Dr (mult D R)
          J (/ (sumsq Dr) 2)
          X-grad (mmult Dr Theta)
          Theta-grad (mmult (trans Dr) X)]
      {:cost (+ J (* (+ (sumsq Theta) (sumsq X)) lambda 0.5))
       :grad [(plus X-grad (mult X lambda)) (plus Theta-grad (mult Theta lambda))]})))

(defn- normalize-ratings [Y]
  (let [Ymean (vec (map #(mean (filter pos? %)) Y))]
    [(matrix (map minus Y Ymean)) Ymean]))

(defn- normal-matrix [rows cols]
  (matrix (sample-normal (* rows cols)) cols))

(defn learn-movie-ratings [Y R new-ratings lambda]
  (let [[num-movies _] (dim Y)
        ratings (reduce (fn [v [m r]] (assoc v m r)) (zeroes num-movies) (partition 2 new-ratings))
        Y (bind-columns ratings Y)
        R (bind-columns (vec (indicator pos? ratings)) R)
        [_ num-users] (dim Y)
        num-features 10

        [Ynorm Ymean] (normalize-ratings Y)
        inits [(normal-matrix num-movies num-features) (normal-matrix num-users num-features)]
        [X Theta] (fmincg
                    (cofi-cost-fn Ynorm R lambda) inits
                    :reshape (reshape inits) :max-iter 100 :verbose true)]
    (vec (map + (sel (mmult X (trans Theta)) :cols 0) Ymean))))

(time
  (if *command-line-args*
    (let [my-ratings [0 4, 6 3, 11 5, 53 4, 63 5, 65 3, 68 5, 97 2, 182 4, 225 5, 354 5]
          {:keys [Y R]} (init-ex8-cofi)
          lambda 10
          predictions (learn-movie-ratings Y R my-ratings lambda)
          movies (reduce conj [] (map #(second (.split % " " 2)) (line-seq (reader "data/movie_ids.txt"))))
          recommendations (map (fn [[idx score]] [(movies idx) score]) (sort-by second > (map-indexed vector predictions)))]
      (doall (map println (take 10 recommendations))))))
