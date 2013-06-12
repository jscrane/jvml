(ns kaggle.titanic.data
  (:use (incanter core io stats)))

(defn- sex [pass] (assoc pass :sex (if (= (:sex pass) "male") 1 0)))

(defn- embarked [pass]
  (let [emb (:embarked pass)
        embarked? (if (= "" emb) 0 1)]
    (assoc pass :embarked? embarked? :embarked (cond (= emb "C") 0 (= emb "S") 1 (= emb "Q") 2 :else -1))))

(defn- fare [pass]
  (let [f (:fare pass)
        fare? (if (or (= "" f) (zero? f)) 0 1)]
    (assoc pass :fare? fare? :fare (if (= "" f) 0 f))))

(defn- age [pass]
  (let [a (:age pass)
        age? (if (= "" a) 0 1)]
    (assoc pass :age? age? :age (if (= "" a) 0 a))))

(defn- cleanup-classifiers [passengers]
  (map (comp sex embarked fare age) passengers))

(defn- most-common-port [passengers]
  (let [ports (map :embarked (filter (comp pos? :embarked? ) passengers))]
    (int (median ports))))

(defn- compute-medians [passengers f keys]
  (reduce (fn [m k] (assoc m k (f passengers k))) {} keys))

(defn- median-fare [passengers {pclass :pclass embarked :embarked}]
  (let [fares (map :fare (filter #(and (= pclass (:pclass %)) (= embarked (:embarked %))) passengers))]
    (median fares)))

(defn- median-fares [passengers]
  (compute-medians passengers median-fare (for [e [0 1 2] c [1 2 3]] {:pclass c :embarked e})))

(defn- median-age [passengers {pclass :pclass sex :sex}]
  (let [ages (map :age (filter #(and (= pclass (:pclass %)) (= sex (:sex %))) passengers))]
    (median ages)))

(defn- median-ages [passengers]
  (compute-medians passengers median-age (for [c [1 2 3] s [0 1]] {:sex s :pclass c})))

(defn- missing-port [port passengers]
  (map #(if (pos? (:embarked? %)) % (assoc % :embarked port)) passengers))

(defn- missing-fare [fares passengers]
  (map #(if (pos? (:fare? %)) % (assoc % :fare (fares (select-keys % [:pclass :embarked])))) passengers))

(defn- missing-age [ages passengers]
  (map #(if (pos? (:age? %)) % (assoc % :age (ages (select-keys % [:pclass :sex])))) passengers))

(defn- read-csv [file]
  (second (second (read-dataset file :header true))))

(defn init [m-val interesting-keys]
  (let [training-data (cleanup-classifiers (read-csv "src/main/clojure/kaggle/titanic/train.csv"))
        test-data (cleanup-classifiers (read-csv "src/main/clojure/kaggle/titanic/test.csv"))
        all-data (concat training-data test-data)

        port (partial missing-port (most-common-port all-data))
        fare (partial missing-fare (median-fares all-data))
        age (partial missing-age (median-ages all-data))

        training (shuffle (-> training-data port fare age))
        test (-> test-data port fare age)
        train-y (map :survived training)
        train-X (map #(vec (vals (select-keys % interesting-keys))) training)
        test-X (map #(vec (vals (select-keys % interesting-keys))) test)]
    {:training training :test test
      :train-y train-y :train-X train-X
      :y (vec (drop m-val train-y)) :yval (vec (take m-val train-y))
      :X (matrix (drop m-val train-X)) :Xval (matrix (take m-val train-X))
     :Xtest (matrix test-X)}))

(defn submit [predictions]
  (spit "submission.txt" (apply str (map #(str % "\n") predictions))))