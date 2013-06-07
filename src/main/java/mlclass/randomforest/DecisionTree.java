package mlclass.randomforest;

import cern.colt.matrix.tdouble.DoubleMatrix1D;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * Creates a decision tree based on the specifications of random forest trees
 * Originally from http://randomforestadk.cvs.sourceforge.net/
 *
 * @author kapelner
 * @see <a href="http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm">Breiman's Random Forests (UC Berkeley)</a>
 */
public final class DecisionTree {

    /**
     * Instead of checking each index we'll skip every INDEX_SKIP indices unless there's less than MIN_SIZE_TO_CHECK_EACH
     */
    private static final int INDEX_SKIP = 3;

    /**
     * If there's less than MIN_SIZE_TO_CHECK_EACH points, we'll check each one
     */
    private static final int MIN_SIZE_TO_CHECK_EACH = 10;

    /**
     * If the number of data points is less than MIN_NODE_SIZE, we won't continue splitting, we'll take the majority vote
     */
    private static final int MIN_NODE_SIZE = 5;

    /**
     * the number of data records
     */
    private final int N;

    /**
     * the number of samples left out of the bootstrap of all N to test error rate
     *
     * @see <a href="http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#ooberr">OOB error estimate</a>
     */
    public final int testN;

    /**
     * Of the testN, the number that were correctly identified
     */
    public final int correct;

    /**
     * an estimate of the importance of each attribute in the data record
     *
     * @see <a href="http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#varimp>Variable Importance</a>
     */
    public final int[] importances;

    /**
     * This is the root of the Decision Tree
     */
    private final TreeNode root;

    /**
     * The Random Forest this decision tree belongs to
     */
    private final RandomForest forest;

    /**
     * Construct a decision tree from a data matrix.
     * It first creates a bootstrap sample, the train data matrix, as well as the left out records,
     * the test data matrix. Then it creates the tree, then calculates the variable importances (not essential)
     * and then removes the links to the actual data (to save memory)
     *
     * @param data   The data matrix as a List of int arrays - each array is one record, each index in the array is one attribute, and the last index is the class
     *               (ie [ x1, x2, . . ., xM, Y ]).
     * @param forest The random forest this decision tree belongs to
     */
    public DecisionTree(List<DoubleMatrix1D> data, RandomForest forest) {
        this.forest = forest;
        N = data.size();
        importances = new int[forest.M];

        System.out.println("\nMake a Dtree N:" + N + " M:" + forest.M + " Ms:" + forest.Ms);

        List<DoubleMatrix1D> train = new ArrayList<DoubleMatrix1D>(); //data becomes the "bootstrap" - that's all it knows
        List<DoubleMatrix1D> test = new ArrayList<DoubleMatrix1D>();

        bootstrapSample(data, train, test);
        this.testN = test.size();

        this.root = new TreeNode();
        root.data = train;
        recursiveSplit(root);
        System.out.println("\ndone split");

        int correct = 0;
        for (DoubleMatrix1D record : test) {
            int Class = evaluate(record);
            forest.updateOOBEstimate(record, Class);
            if (Class == getClass(record))
                correct++;
        }

        double err = 1 - correct / ((double) test.size());
        System.out.println("of left out data, error rate:" + err);
        this.correct = correct;

        for (int m = 0; m < forest.M; m++) {
            List<DoubleMatrix1D> data1 = randomlyPermuteAttribute(copyData(test), m);
            int correctAfterPermute = 0;
            for (DoubleMatrix1D arr : data1) {
                int prediction = evaluate(arr);
                if (prediction == getClass(arr))
                    correctAfterPermute++;
            }
            importances[m] += (correct - correctAfterPermute);
        }
        flushData(root);
    }

    /**
     * This will classify a new data record by using tree
     * recursion and testing the relevant variable at each node.
     * <p/>
     * This is probably the most-used function in all of <b>GemIdent</b>.
     * It would make sense to inline this in assembly for optimal performance.
     *
     * @param record the data record to be classified
     * @return the class the data record was classified into
     */
    public int evaluate(DoubleMatrix1D record) {
        TreeNode evalNode = root;

        while (true) {
            if (evalNode.isLeaf)
                return evalNode.Class;
            if (record.getQuick(evalNode.splitAttributeM) <= evalNode.splitValue)
                evalNode = evalNode.left;
            else
                evalNode = evalNode.right;
        }
    }

    /**
     * Takes a list of data records, and switches the mth attribute across data records.
     * This is important in order to test the importance of the attribute. If the attribute
     * is randomly permuted and the result of the classification is the same, the attribute is
     * not important to the classification and vice versa.
     *
     * @param test The data matrix to be permuted
     * @param m    The attribute index to be permuted
     * @return The data matrix with the mth column randomly permuted
     * @see <a href="http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#varimp">Variable Importance</a>
     */
    private List<DoubleMatrix1D> randomlyPermuteAttribute(List<DoubleMatrix1D> test, int m) {
        int num = test.size() * 2;
        for (int i = 0; i < num; i++) {
            int a = (int) Math.floor(Math.random() * test.size());
            int b = (int) Math.floor(Math.random() * test.size());
            DoubleMatrix1D arrA = test.get(a);
            DoubleMatrix1D arrB = test.get(b);
            double temp = arrA.getQuick(m);
            arrA.setQuick(m, arrB.getQuick(m));
            arrB.setQuick(m, temp);
        }
        return test;
    }

    /**
     * Creates a copy of the data matrix
     *
     * @param data the data matrix to be copied
     * @return the cloned data matrix
     */
    private List<DoubleMatrix1D> copyData(List<DoubleMatrix1D> data) {
        List<DoubleMatrix1D> copy = new ArrayList<DoubleMatrix1D>(data.size());
        for (DoubleMatrix1D d : data)
            copy.add(d.copy());

        return copy;
    }

    /**
     * @author kapelner
     */
    private class TreeNode {
        public boolean isLeaf;
        public TreeNode left;
        public TreeNode right;
        public int splitAttributeM = -99;
        public Integer Class;
        public List<DoubleMatrix1D> data;
        public int splitValue = -99;
        public int generation = 1;
    }

    private class DoubleHolder {
        public double d;

        public DoubleHolder(double d) {
            this.d = d;
        }
    }

    /**
     * This is the crucial function in tree creation.
     * <p/>
     * <ul>
     * <li>Step A
     * Check if this node is a leaf, if so, it will mark isLeaf true
     * and mark Class with the leaf's class. The function will not
     * recurse past this point.
     * </li>
     * <li>Step B
     * Create a left and right node and keep their references in
     * this node's left and right fields. For debugging purposes,
     * the generation number is also recorded. The {@link RandomForest#Ms Ms} attributes are
     * now chosen by the {@link #getVarsToInclude() getVarsToInclude} function
     * </li>
     * <li>Step C
     * For all Ms variables, first {@link #sortAtAttribute(List, int) sort} the data records by that attribute
     * , then look through the values from lowest to
     * highest. If value i is not equal to value i+1, record i in the list of "indicesToCheck."
     * This speeds up the splitting. If the number of indices in indicesToCheck >  MIN_SIZE_TO_CHECK_EACH
     * then we will only {@link #checkPosition(int, int, int, DoubleHolder, DecisionTree.TreeNode) check} the
     * entropy at every {@link #INDEX_SKIP INDEX_SKIP} index otherwise, we {@link #checkPosition(int, int, int, DecisionTree.DoubleHolder, DecisionTree.TreeNode) check}
     * the entropy for all. The "E" variable records the entropy and we are trying to find the minimum in which to split on
     * </li>
     * <li>Step D
     * The newly generated left and right nodes are now checked:
     * If the node has only one record, we mark it as a leaf and set its class equal to
     * the class of the record. If it has less than {@link #MIN_NODE_SIZE MIN_NODE_SIZE}
     * records, then we mark it as a leaf and set its class equal to the {@link #getMajorityClass(List) majority class}.
     * If it has more, then we do a manual check on its data records and if all have the same class, then it
     * is marked as a leaf. If not, then we run recursiveSplit on
     * that node
     * </li>
     * </ul>
     *
     * @param parent The node of the parent
     */
    private void recursiveSplit(TreeNode parent) {


        if (!parent.isLeaf) {

            //-------------------------------Step A
            Integer Class = checkIfLeaf(parent.data);
            if (Class != null) {
                parent.isLeaf = true;
                parent.Class = Class;
                return;
            }

            //-------------------------------Step B
            int Nsub = parent.data.size();

            parent.left = new TreeNode();
            parent.left.generation = parent.generation + 1;
            parent.right = new TreeNode();
            parent.right.generation = parent.generation + 1;

            List<Integer> vars = getVarsToInclude();

            DoubleHolder lowestE = new DoubleHolder(Double.MAX_VALUE);

            //-------------------------------Step C
            for (int m : vars) {

                sortAtAttribute(parent.data, m);

                List<Integer> indicesToCheck = new ArrayList<Integer>();
                for (int n = 1; n < Nsub; n++) {
                    int classA = getClass(parent.data.get(n - 1));
                    int classB = getClass(parent.data.get(n));
                    if (classA != classB)
                        indicesToCheck.add(n);
                }

                if (indicesToCheck.size() == 0) {
                    parent.isLeaf = true;
                    parent.Class = getClass(parent.data.get(0));
                    continue;
                }
                if (indicesToCheck.size() > MIN_SIZE_TO_CHECK_EACH) {
                    for (int i = 0; i < indicesToCheck.size(); i += INDEX_SKIP) {
                        checkPosition(m, indicesToCheck.get(i), Nsub, lowestE, parent);
                        if (lowestE.d == 0)
                            break;
                    }
                } else {
                    for (int n : indicesToCheck) {
                        checkPosition(m, n, Nsub, lowestE, parent);
                        if (lowestE.d == 0)
                            break;
                    }
                }
                if (lowestE.d == 0)
                    break;
            }
            //-------------------------------Step D
            if (parent.left.data.size() == 1) {
                parent.left.isLeaf = true;
                parent.left.Class = getClass(parent.left.data.get(0));
            } else if (parent.left.data.size() < MIN_NODE_SIZE) {
                parent.left.isLeaf = true;
                parent.left.Class = getMajorityClass(parent.left.data);
            } else {
                Class = checkIfLeaf(parent.left.data);
                if (Class == null) {
                    parent.left.isLeaf = false;
                    parent.left.Class = null;
                } else {
                    parent.left.isLeaf = true;
                    parent.left.Class = Class;
                }
            }
            if (parent.right.data.size() == 1) {
                parent.right.isLeaf = true;
                parent.right.Class = getClass(parent.right.data.get(0));
            } else if (parent.right.data.size() < MIN_NODE_SIZE) {
                parent.right.isLeaf = true;
                parent.right.Class = getMajorityClass(parent.right.data);
            } else {
                Class = checkIfLeaf(parent.right.data);
                if (Class == null) {
                    parent.right.isLeaf = false;
                    parent.right.Class = null;
                } else {
                    parent.right.isLeaf = true;
                    parent.right.Class = Class;
                }
            }

            if (!parent.left.isLeaf)
                recursiveSplit(parent.left);
            if (!parent.right.isLeaf)
                recursiveSplit(parent.right);
        }
    }

    /**
     * Given a data matrix, return the most popular Y value (the class)
     *
     * @param data The data matrix
     * @return The most popular class
     */
    private int getMajorityClass(List<DoubleMatrix1D> data) {
        int[] counts = new int[forest.C];
        for (DoubleMatrix1D record : data) {
            int Class = getClass(record);
            counts[Class]++;
        }
        int index = -99;
        int max = Integer.MIN_VALUE;
        for (int i = 0; i < counts.length; i++) {
            if (counts[i] > max) {
                max = counts[i];
                index = i;
            }
        }
        return index;
    }

    /**
     * Checks the {@link #calculateEntropy(double[]) entropy} of an index in a data matrix at a particular attribute (m)
     * and returns the entropy. If the entropy is lower than the minimum to date (lowestE), it is set to the minimum.
     * <p/>
     * The total entropy is calculated by getting the sub-entropy for below the split point and after the split point.
     * The sub-entropy is calculated by first getting the {@link #getClassProbs(List) proportion} of each of the classes
     * in this sub-data matrix. Then the entropy is {@link #calculateEntropy(double[]) calculated}. The lower sub-entropy
     * and upper sub-entropy are then weight averaged to obtain the total entropy.
     *
     * @param m       the attribute to split on
     * @param n       the index to check
     * @param Nsub    the number of records in the data matrix
     * @param lowestE the minimum entropy to date
     * @param parent  the parent node
     * @return the entropy of this split
     */
    private double checkPosition(int m, int n, int Nsub, DoubleHolder lowestE, TreeNode parent) {

        if (n < 1) //exit conditions
            return 0;
        if (n > Nsub)
            return 0;

        List<DoubleMatrix1D> lower = getLower(parent.data, n);
        List<DoubleMatrix1D> upper = getUpper(parent.data, n);
        if (lower == null)
            System.out.println("lower list null");
        if (upper == null)
            System.out.println("upper list null");
        double[] pl = getClassProbs(lower);
        double[] pu = getClassProbs(upper);
        double eL = calculateEntropy(pl);
        double eU = calculateEntropy(pu);

        double e = (eL * lower.size() + eU * upper.size()) / ((double) Nsub);
        if (e < lowestE.d) {
            lowestE.d = e;
            parent.splitAttributeM = m;
            parent.splitValue = (int)parent.data.get(n).getQuick(m);
            parent.left.data = lower;
            parent.right.data = upper;
        }
        return e;
    }

    /**
     * Given a data record, return the Y value - take the last index
     *
     * @param record the data record
     * @return its y value (class)
     */
    private int getClass(DoubleMatrix1D record) {
        return (int)record.getQuick(forest.M);
    }

    /**
     * Given a data matrix, check if all the y values are the same. If so,
     * return that y value, null if not
     *
     * @param data the data matrix
     * @return the common class (null if not common)
     */
    private Integer checkIfLeaf(List<DoubleMatrix1D> data) {
        boolean isLeaf = true;
        int ClassA = getClass(data.get(0));
        for (int i = 1; i < data.size(); i++) {
            DoubleMatrix1D recordB = data.get(i);
            if (ClassA != getClass(recordB)) {
                isLeaf = false;
                break;
            }
        }
        if (isLeaf)
            return getClass(data.get(0));

        return null;
    }

    /**
     * Split a data matrix and return the upper portion
     *
     * @param data   the data matrix to be split
     * @param nSplit return all data records above this index in a sub-data matrix
     * @return the upper sub-data matrix
     */
    private List<DoubleMatrix1D> getUpper(List<DoubleMatrix1D> data, int nSplit) {
        int N = data.size();
        List<DoubleMatrix1D> upper = new ArrayList<DoubleMatrix1D>(N - nSplit);
        for (int n = nSplit; n < N; n++)
            upper.add(data.get(n));
        return upper;
    }

    /**
     * Split a data matrix and return the lower portion
     *
     * @param data   the data matrix to be split
     * @param nSplit return all data records below this index in a sub-data matrix
     * @return the lower sub-data matrix
     */
    private List<DoubleMatrix1D> getLower(List<DoubleMatrix1D> data, int nSplit) {
        List<DoubleMatrix1D> lower = new ArrayList<DoubleMatrix1D>(nSplit);
        for (int n = 0; n < nSplit; n++)
            lower.add(data.get(n));
        return lower;
    }

    /**
     * This class compares two data records by numerically comparing a specified attribute
     *
     * @author kapelner
     */
    private final class AttributeComparator implements Comparator {
        /**
         * the specified attribute
         */
        private int m;

        /**
         * Create a new comparator
         *
         * @param m the attribute in which to compare on
         */
        public AttributeComparator(int m) {
            this.m = m;
        }

        /**
         * Compares two data records.
         *
         * @param o1 data record A
         * @param o2 data record B
         * @return -1 if A[m] < B[m], 1 if A[m] > B[m], 0 if equal
         */
        public int compare(Object o1, Object o2) {
            double a = ((DoubleMatrix1D) o1).getQuick(m);
            double b = ((DoubleMatrix1D) o2).getQuick(m);
            if (a < b)
                return -1;
            if (a > b)
                return 1;
            return 0;
        }
    }

    /**
     * Sorts a data matrix by an attribute from lowest record to highest record
     *
     * @param data the data matrix to be sorted
     * @param m    the attribute to sort on
     */
    @SuppressWarnings("unchecked")
    private void sortAtAttribute(List<DoubleMatrix1D> data, int m) {
        Collections.sort(data, new AttributeComparator(m));
    }

    /**
     * Given a data matrix, return a probability mass function representing
     * the frequencies of a class in the matrix (the y values)
     *
     * @param records the data matrix to be examined
     * @return the probability mass function
     */
    private double[] getClassProbs(List<DoubleMatrix1D> records) {

        double N = records.size();

        int[] counts = new int[forest.C];

        for (DoubleMatrix1D record : records)
            counts[getClass(record)]++;

        double[] ps = new double[forest.C];
        for (int c = 0; c < forest.C; c++)
            ps[c] = counts[c] / N;
        return ps;
    }

    /**
     * ln(2)
     */
    private static final double logoftwo = Math.log(2);

    /**
     * Given a probability mass function indicating the frequencies of
     * class representation, calculate an "entropy" value using the method
     * in Tan Steinbach Kumar's "Data Mining" textbook
     *
     * @param ps the probability mass function
     * @return the entropy value calculated
     */
    private double calculateEntropy(double[] ps) {
        double e = 0;
        for (double p : ps) {
            if (p != 0) //otherwise it will divide by zero - see TSK p159
                e += p * Math.log(p) / logoftwo;
        }
        return -e; //according to TSK p158
    }

    /**
     * Of the M attributes, select {@link RandomForest#Ms Ms} at random.
     *
     * @return The list of the Ms attributes' indices
     */
    private List<Integer> getVarsToInclude() {
        boolean[] whichVarsToInclude = new boolean[forest.M];
        while (true) {
            int a = (int) Math.floor(Math.random() * forest.M);
            whichVarsToInclude[a] = true;
            int N = 0;
            for (int i = 0; i < forest.M; i++)
                if (whichVarsToInclude[i])
                    N++;
            if (N == forest.Ms)
                break;
        }

        List<Integer> shortRecord = new ArrayList<Integer>(forest.Ms);
        for (int i = 0; i < forest.M; i++)
            if (whichVarsToInclude[i])
                shortRecord.add(i);
        return shortRecord;
    }

    /**
     * Create a boostrap sample of a data matrix
     *
     * @param data  the data matrix to be sampled
     * @param train the bootstrap sample
     * @param test  the records that are absent in the bootstrap sample
     */
    private void bootstrapSample(List<DoubleMatrix1D> data, List<DoubleMatrix1D> train, List<DoubleMatrix1D> test) {
        List<Integer> indices = new ArrayList<Integer>(N);
        for (int n = 0; n < N; n++)
            indices.add((int) Math.floor(Math.random() * N));
        List<Boolean> in = new ArrayList<Boolean>(N);
        for (int n = 0; n < N; n++)
            in.add(false); //have to initialize it first
        for (int num : indices) {
            train.add((data.get(num)).copy());
            in.set(num, true);
        }
        for (int i = 0; i < N; i++)
            if (!in.get(i))
                test.add((data.get(i)).copy());
    }

    /**
     * Recursively deletes all data records from the tree. This is run after the tree
     * has been computed and can stand alone to classify incoming data.
     *
     * @param node initially, the root node of the tree
     */
    private void flushData(TreeNode node) {
        node.data = null;
        if (node.left != null)
            flushData(node.left);
        if (node.right != null)
            flushData(node.right);
    }
}