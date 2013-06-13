(ns kaggle.titanic.classifiers)

;
; mapping strings to integers and flagging missing fields (fare, age and embarked)
;
(defn- sex [pass] (assoc pass :sex (if (= (:sex pass) "male") 1 0)))

(defn- embarked [{e :embarked :as pass}]
  (let [embarked? (if (= "" e) 0 1)]
    (assoc pass :embarked? embarked? :embarked (cond (= e "C") 0 (= e "S") 1 (= e "Q") 2 :else -1))))

(defn- fare [{f :fare s :sibsp p :parch :as pass}]
  (let [fare (if (= "" f) 0 f)
        fare? (if (zero? fare) 0 1)
        family (inc (+ s p))]
    (assoc pass :fare? fare? :fare (/ fare family) :family family)))

(defn- age [{a :age :as pass}]
  (let [age (if (= "" a) 0 a)
        age? (if (zero? age) 0 1)]
    (assoc pass :age? age? :age age)))

(defn- cabin [{c :cabin :as pass}]
  (assoc pass :cabin? (if (= "" c) 0 1)))

; "Beckwith, Mrs. Richard Leonard (Sallie Monypeny)"
(defn- name-parts [name]
  (let [munge-re #"([^,]+),\s+([^.]+)\.\s+(\S+)\s*(?:.*?)?(?:\s*\(.*\s+([^)]+)\))?"
        parts (rest (re-matches munge-re name))]
    (zipmap [:last :title :first :unmarried ] parts)))

(defn- title [{name :name :as pass}]
  (let [t (:title (name-parts name))]
    (assoc pass :title (cond (= t "Mr") 1 (= t "Mrs") 2 (= t "Master") 3 (= t "Miss") 4 :else 0))))

(defn cleanup-classifiers [passengers]
  (map (comp sex embarked fare age cabin title) passengers))

