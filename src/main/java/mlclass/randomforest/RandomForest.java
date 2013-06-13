package mlclass.randomforest;

import cern.colt.matrix.tdouble.DoubleMatrix1D;

import java.util.*;
import java.util.concurrent.*;

/**
 * Houses a Breiman Random Forest
 * Originally from http://randomforestadk.cvs.sourceforge.net/
 *
 * @author kapelner
 * @see <a href="http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm">Breiman's Random Forests (UC Berkeley)</a>
 */
public final class RandomForest {

    public final int C;

    public final int M;

    /**
     * Of the M total attributes, the random forest computation requires a subset of them
     * to be used and picked via random selection. "Ms" is the number of attributes in this
     * subset. The formula used to generate Ms was recommended on Breiman's website.
     */
    public final int Ms;

    /**
     * the collection of the forest's decision trees
     */
    private final Collection<DecisionTree> trees;

    /**
     * this is an array whose indices represent the forest-wide importance for that given attribute
     */
    public final double[] importances;

    /**
     * This maps from a data record to an array that records the classifications by the trees where it was a "left out" record (the indices are the class and the values are the counts)
     */
    public final Map<DoubleMatrix1D, int[]> estimateOOB;

    /**
     * the total forest-wide error
     */
    public final double error;

    /**
     * Constructs a Breiman random forest
     *
     * @param numTrees the number of trees in the forest
     * @param C        the number of categorical responses of the data (the classes, the "Y" values)
     * @param M        the number of attributes in the data
     * @param data     the training data
     */
    public RandomForest(int numTrees, int C, int M, List<DoubleMatrix1D> data) {
        this.C = C;
        this.M = M;
        this.Ms = (int) Math.round(Math.log(M) / Math.log(2) + 1);   //recommended by Breiman
        this.trees = new ConcurrentLinkedQueue<DecisionTree>();
        this.estimateOOB = new ConcurrentHashMap<DoubleMatrix1D, int[]>(data.size());

        ExecutorService treePool = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        for (int t = 0; t < numTrees; t++)
            treePool.execute(new CreateTree(data, this));
        treePool.shutdown();
        try {
            treePool.awaitTermination(Long.MAX_VALUE, TimeUnit.SECONDS); //effectively infinity
        } catch (InterruptedException ignored) {
            System.out.println("interrupted exception in Random Forests");
        }

        /**
         * This calculates the forest-wide error rate. For each "left out"
         * data record, if the class with the maximum count is equal to its actual
         * class, then it's been predicted correctly.
         */
        double N = 0;
        int correct = 0;
        for (DoubleMatrix1D record : estimateOOB.keySet()) {
            N++;
            int[] map = estimateOOB.get(record);
            int Class = findMaxIndex(map);
            if (Class == (int) record.getQuick(M))
                correct++;
        }
        error = 1 - correct / N;

        /**
         * This calculates the forest-wide importance levels for all attributes.
         */
        importances = new double[M];
        for (DecisionTree tree : trees) {
            for (int i = 0; i < M; i++)
                importances[i] += tree.importances[i];
        }
        for (int i = 0; i < M; i++)
            importances[i] /= numTrees;
    }

    /**
     * Update the error map by recording a class prediction
     * for a given data record
     *
     * @param record the data record classified
     * @param Class  the class
     */
    public void updateOOBEstimate(DoubleMatrix1D record, int Class) {
        if (estimateOOB.get(record) == null) {
            int[] map = new int[C];
            map[Class]++;
            estimateOOB.put(record, map);
        } else {
            int[] map = estimateOOB.get(record);
            map[Class]++;
        }
    }

    /**
     * This class houses the machinery to generate one decision tree in a thread pool environment.
     *
     * @author kapelner
     */
    private final class CreateTree implements Runnable {
        /**
         * the training data to generate the decision tree (same for all trees)
         */
        private final List<DoubleMatrix1D> data;
        /**
         * the current forest
         */
        private final RandomForest forest;

        public CreateTree(List<DoubleMatrix1D> data, RandomForest forest) {
            this.data = data;
            this.forest = forest;
        }

        /**
         * Creates the decision tree
         */
        public void run() {
            trees.add(new DecisionTree(data, forest));
        }
    }

    /**
     * Evaluates an incoming data record.
     * It first allows all the decision trees to classify the record,
     * then it returns the majority vote
     *
     * @param record the data record to be classified
     */
    public int evaluate(DoubleMatrix1D record) {
        int[] counts = new int[C];
        for (DecisionTree tree: trees)
            counts[tree.evaluate(record)]++;

        return findMaxIndex(counts);
    }

    /**
     * Given an array, return the index that houses the maximum value
     *
     * @param arr the array to be investigated
     * @return the index of the greatest value in the array
     */
    private static int findMaxIndex(int[] arr) {
        int index = 0;
        int max = Integer.MIN_VALUE;
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] > max) {
                max = arr[i];
                index = i;
            }
        }
        return index;
    }
}
