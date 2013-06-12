(ns kaggle.titanic
  (:import (mlclass.randomforest RandomForest))
  (:use (incanter core io charts)
        (ml util optim logistic svm)))

; "Beckwith, Mrs. Richard Leonard (Sallie Monypeny)"
(def munge-re #"([^,]+),\s+([^.]+)\.\s+(\S+)\s*(?:.*?)?(?:\s*\(.*\s+([^)]+)\))?")

(defn name-parts [datum]
  (let [parts (rest (re-matches munge-re (:name datum)))]
    (zipmap [:surname :title :own-or-husband-first-name :maiden-name ] parts)))

; the most common port is "S"
(defn cleanup [row]
  (let [g (if (= (:sex row) "male") 0 1)
        e (:embarked row)
        em (if (= "" e) "S" e)
        f (:fare row)
        fa (if (= "" f) 0 f)
        p (cond
            (= em "C") -1
            (= em "Q") 1
            (= em "S") 0)
        np (name-parts row)
        ]
    (merge (assoc row :sex g :embarked p :fare fa) np)))

(def training-data (map cleanup (second (second (read-dataset "kaggle/titanic-train.csv" :header true)))))
(def test-data (map cleanup (second (second (read-dataset "kaggle/titanic-test.csv" :header true)))))
(def all-data (concat training-data test-data))
(def data all-data)

(def name-counts (reduce #(assoc %1 %2 (inc (get %1 %2 0))) {} (map #(select-keys % [:surname :own-or-husband-first-name ]) all-data)))

(def ticket-passengers
  (reduce (fn [tickets passenger]
            (let [ticket (:ticket passenger)]
              (assoc tickets ticket (conj (get tickets ticket #{}) passenger))))
    {} data))

(def ticket-fares
  (reduce (fn [tickets passengers]
            (let [f (first passengers) c (count passengers)]
              (assoc tickets (:ticket f)
                {:fare (double (/ (:fare f) c)) :pclass (:pclass f) :embarked (:embarked f)})))
    {} (vals ticket-passengers)))

(def class-port-fares
  (reduce (fn [class-port-fares ticket-fare]
            (let [key (dissoc ticket-fare :fare )
                  fare (:fare ticket-fare)
                  cpf (class-port-fares key)]
              (if (zero? fare)
                class-port-fares
                (assoc class-port-fares key (if cpf [(+ fare (first cpf)) (inc (second cpf)) (max fare (nth cpf 2))] [fare 1 fare])))))
    {} (vals ticket-fares)))

(def class-port-sex-ages
  (reduce (fn [class-port-sex-ages passenger]
            (let [{:keys [pclass embarked sex age]} passenger
                  key {:pclass pclass :embarked embarked :sex sex}
                  cpa (class-port-sex-ages key)]
              (if (or (= age "") (zero? age))
                class-port-sex-ages
                (assoc class-port-sex-ages key (if cpa [(+ age (first cpa)) (inc (second cpa)) (max age (nth cpa 2))] [age 1 age])))))
    {} data))

(defn split-siblings-spouse-parent-child [passenger]
  (let [c (name-counts (select-keys passenger [:surname :own-or-husband-first-name ]))
        sibsp (:sibsp passenger)
        parch (:parch passenger)
        title (:title passenger)
        with-spouse (dec c)
        siblings (- sibsp with-spouse)
        blah (cond
               (zero? parch) {:children 0 :parents 0}
               (pos? with-spouse) {:children parch :parents 0}
               (pos? siblings) {:children 0 :parents parch}
               (= "Master" title) {:children 0 :parents parch}
               (= "Miss" title) {:children 0 :parents parch}
               (= "Mrs" title) {:children parch :parents 0}
               :else (println passenger)
               )
        ]
    (merge {:with-spouse with-spouse :siblings siblings} blah)))

(defn add-missing [passenger ppt]
  (let [{:keys [age fare pclass embarked sex ticket]} passenger
        [a na ma] (class-port-sex-ages {:pclass pclass :embarked embarked :sex sex})
        [f nf mf] (class-port-fares {:pclass pclass :embarked embarked})
        aage (/ a na) afare (/ f nf)]
      (split-siblings-spouse-parent-child (assoc passenger
                                            ;      :age (/ (if (= age "") (/ a na) age) ma)
                                            ;      :fare (/ (if (= fare "") (/ f nf) fare) mf)
                                            :norm-age (if (= age "") 1 (/ age aage))
                                            :norm-fare (if (= fare "") 1 (/ fare afare))
                                            :companions (ppt ticket)))))

(defn passengers-per-ticket [data]
  (reduce (fn [ppt passenger]
            (let [t (:ticket passenger) c (ppt t)]
              (assoc ppt t (if c (inc c) 1)))) {} data))

(defn interest [data]
  (let [ppt (passengers-per-ticket data)]
    (map
      ;#(select-keys (add-missing % ppt) [:age :norm-age :sex :survived :pclass :siblings :with-spouse :parch :fare :norm-fare :embarked :companions :name :surname :cabin])
      #(add-missing % ppt)
      data)))

(defn grep
  ([pred key value data]
    (filter #(pred (key %) value) data))
  ([key value data]
    (grep = key value data)))
