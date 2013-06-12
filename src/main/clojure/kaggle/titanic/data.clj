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

(defn- median-fare [passengers {pclass :pclass embarked :embarked}]
  (let [fares (map :fare (filter #(and (= pclass (:pclass %)) (= embarked (:embarked %))) passengers))]
    (median fares)))

(defn- compute-medians [passengers f keys]
  (reduce (fn [m k] (assoc m k (f passengers k))) {} keys))

(defn- median-fares [passengers]
  (compute-medians passengers median-fare (for [e [0 1 2] c [1 2 3]] {:pclass c :embarked e})))

(defn- median-age [passengers {pclass :pclass sex :sex}]
  (let [ages (map :age (filter #(and (= pclass (:pclass %)) (= sex (:sex %))) passengers))]
    (median ages)))

(defn- median-ages [passengers]
  (compute-medians passengers median-age (for [c [1 2 3] s [0 1]] {:sex s :pclass c})))

(defn- replace-missing-port [port passengers]
  (map #(if (pos? (:embarked? %)) % (assoc % :embarked port)) passengers))

(defn- replace-missing-fare [fares passengers]
  (map #(if (pos? (:fare? %)) % (assoc % :fare (fares (select-keys % [:pclass :embarked])))) passengers))

(defn- replace-missing-age [ages passengers]
  (map #(if (pos? (:age? %)) % (assoc % :age (ages (select-keys % [:pclass :sex])))) passengers))

(defn init [m-train interesting-keys]
  (let [training-data (cleanup-classifiers (second (second (read-dataset "src/main/clojure/kaggle/titanic/train.csv" :header true))))
        test-data (cleanup-classifiers (second (second (read-dataset "src/main/clojure/kaggle/titanic/test.csv" :header true))))
        all-data (concat training-data test-data)

        port (partial replace-missing-port (most-common-port all-data))
        fare (partial replace-missing-fare (median-fares all-data))
        age (partial replace-missing-age (median-ages all-data))
        training (shuffle (-> training-data port fare age))
        test (-> test-data port fare age)

        all-y (map :survived training)
        all-X (map #(vec (vals (select-keys % interesting-keys))) training)
        test-X (map #(vec (vals (select-keys % interesting-keys))) test)]
    {:training training :test test
      :all-y all-y :all-X all-X
      :y (vec (take m-train all-y)) :yval (vec (drop m-train all-y))
      :X (matrix (take m-train all-X)) :Xval (matrix (drop m-train all-X))
     :Xtest (matrix test-X)}))