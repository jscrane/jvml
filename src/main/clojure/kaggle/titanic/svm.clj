(ns kaggle.titanic.svm
  (:use
    [ml.svm :only (optimal-model svm-predict)]
    [kaggle.titanic.data :only (init submit)]))

(let [{:keys [y yval X Xval Xtest]} (init 50 #{:age :age? :sex :pclass :sibsp :parch :fare :fare? :embarked :embarked?})
      opt (optimal-model X y Xval yval [0.01 0.03 0.1 0.3 1 3 10 30 100])]

  (println "validation accuracy: " (double (:accuracy opt)) (:C opt) (:sigma opt))
  (submit (svm-predict (:model opt) Xtest)))

