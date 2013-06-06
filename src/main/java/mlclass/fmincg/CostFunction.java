package mlclass.fmincg;

import cern.colt.matrix.tdouble.DoubleMatrix1D;

public interface CostFunction {

    Tuple<Double, DoubleMatrix1D> evaluateCost(DoubleMatrix1D theta);

}
