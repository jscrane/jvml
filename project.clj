(defproject jvml "0.4"
  :repositories [["dev.davidsoergel.com releases" "http://dev.davidsoergel.com/artifactory/repo"]]
  :dependencies [[org.clojure/clojure "1.5.1"]
                 [incanter/incanter-core "1.4.1"]
                 [incanter/incanter-charts "1.4.1"]
                 [net.sourceforge.jmatio/jmatio "1.0"]
                 [edu.berkeley.compbio/jlibsvm "0.902"]
                 [gov.sandia.foundry/porter-stemmer "1.4"]
                 [seesaw "1.4.3"]]
  :plugins [[lein-idea "1.0.1"]]
  :source-paths ["src/main/clojure"]
  :java-source-paths ["src/main/java"]
  :test-paths ["test"])
