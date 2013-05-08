(defproject jvml "0.2"
  :repositories [["local" ~(str (.toURI (java.io.File. "maven_repository")))]
                 ["dev.davidsoergel.com releases" "http://dev.davidsoergel.com/artifactory/repo"]]
  :dependencies [[org.clojure/clojure "1.4.0"]
                 [incanter/incanter-core "1.4.1"]
                 [incanter/incanter-charts "1.4.1"]
                 [com.jmatio/jmatio "071005"]
                 [edu.berkeley.compbio/jlibsvm "0.902"]
                 [gov.sandia.foundry/porter-stemmer "1.4"]])