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
        [X Theta] (fmincg
                    (cofi-cost-fn Ynorm R lambda)
                    [(normal-matrix num-movies num-features) (normal-matrix num-users num-features)]
                    :max-iter 100 :verbose true)]
    (vec (map + (sel (mmult X (trans Theta)) :cols 0) Ymean))))

;[They Made Me a Criminal (1939) 5.000000044741122]
;[Saint of Fort Washington, The (1993) 5.000000016299654]
;[Santa with Muscles (1996) 5.000000014453069]
;[Entertaining Angels: The Dorothy Day Story (1996) 5.000000009137338]
;[Marlene Dietrich: Shadow and Light (1996)  4.999999998693655]
;[Prefontaine (1997) 4.999999990145056]
;[Someone Else's America (1995) 4.999999986348401]
;[Star Kid (1997) 4.999999975878698]
;[Aiqing wansui (1994) 4.999999968434848]
;[Great Day in Harlem, A (1994) 4.999997619647857]
;"Elapsed time: 5178185.169068 msecs"

(time
  (if *command-line-args*
    (let [my-ratings [0 4, 6 3, 11 5, 53 4, 63 5, 65 3, 68 5, 97 2, 182 4, 225 5, 354 5]
          {:keys [Y R]} (init-ex8-cofi)
          lambda 10
          predictions (learn-movie-ratings Y R my-ratings lambda)
          movies (reduce conj [] (map #(second (.split % " " 2)) (line-seq (reader "data/movie_ids.txt"))))
          recommendations (map (fn [[idx score]] [(movies idx) score]) (sort-by second > (map-indexed vector predictions)))]
      (doall (map println (take 10 recommendations))))))
