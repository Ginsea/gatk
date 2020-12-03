package org.broadinstitute.hellbender.tools.walkers.sv;

import ml.dmlc.xgboost4j.java.*;
import org.apache.commons.math3.util.FastMath;
import org.broadinstitute.barclay.argparser.Argument;
import org.broadinstitute.barclay.argparser.CommandLineProgramProperties;
import org.broadinstitute.barclay.help.DocumentedFeature;
import org.broadinstitute.hellbender.cmdline.programgroups.StructuralVariantDiscoveryProgramGroup;
import org.broadinstitute.hellbender.exceptions.GATKException;

import java.io.*;
import java.util.*;
import java.util.stream.IntStream;

import static org.apache.commons.math3.util.FastMath.*;

@CommandLineProgramProperties(
        summary = "Extract matrix of properties for each variant. Also extract, num_variants x num_trios x 3 tensors of" +
                "allele count and genotype quality. These data will be used to train a variant filter based on min GQ" +
                "(and stratified by other variant properties) that maximizes the admission of variants with Mendelian" +
                "inheritance pattern while omitting non-Mendelian variants." +
                "Derived class must implement abstract method train_filter()",
        oneLineSummary = "Extract data for training min GQ variant filter from .vcf and .ped file with trios.",
        programGroup = StructuralVariantDiscoveryProgramGroup.class
)
@DocumentedFeature
public class XGBoostMinGqVariantFilter extends MinGqVariantFilterBase {
    @Argument(fullName="max-training-rounds", shortName="tr", doc="Maximum number of rounds of training", optional=true, minValue=1)
    public int maxTrainingRounds = 100;
    @Argument(fullName="early-stopping-rounds", shortName="e", doc="Stop training if no improvement is made in validation set for this many rounds. Set <= 0 to disable.", optional=true)
    public int earlyStoppingRounds = 10;
    @Argument(fullName="initial-min-gq-quantile", shortName="q", doc="Initial guess for min GQ, as a quantile of gq in variants of trios.", optional=true)
    public double initialMinGqQuantile = 0.05;
    @Argument(fullName="learning-rate", shortName="lr", doc="Learning rate for xgboost", optional=true)
    public double eta = 1.0;
    @Argument(fullName="max-depth", shortName="d", doc="Max depth of boosted decision tree", optional=true, minValue=1)
    public int maxDepth = 6;
    @Argument(fullName="gamma", doc="Regularization factor for xgboost", optional=true)
    public double gamma = 1.0e-9;
    @Argument(fullName="subsample", doc="Proportion of data selected for each tree", optional=true)
    public double subsample = 0.9;
    @Argument(fullName="colsample-by-tree", doc="Proportion of columns selected for each tree", optional=true)
    public double colsampleByTree = 0.9;
    @Argument(fullName="colsample-by-level", doc="Proportion of columns selected for each level of each tree", optional=true)
    public double colsampleByLevel = 1.0;
    @Argument(fullName="min-child-weight", doc="Proportion of columns selected for each level of each tree", optional=true, minValue=0.0)
    public double minChildWeight = 1.0;
    @Argument(fullName="prediction-scale-factor", doc="Scale factor for raw predictions from xgboost", optional=true)
    public double predictionScaleFactor = 1000.0;
    @Argument(fullName="train-with-targets", doc="Use difference from optimal min GQ as loss, rather than minimizing inheritance/truth loss directly", optional=true)
    public boolean trainWithTargets = true;

    private Booster booster = null;
    private static final String TRAIN_MAT_KEY = "train";
    private static final String VALIDATION_MAT_KEY = "validation";


    protected float[] getRowMajorVariantProperties(int[] rowIndices) {
        if(rowIndices == null) {
            rowIndices = IntStream.range(0, getNumVariants()).toArray();
        }
        final int numRows = rowIndices.length;
        final float[] rowMajorVariantProperties = new float[numRows * getNumProperties()];
        int flatIndex = 0;
        for(final int rowIndex : rowIndices) {
            for(final String propertyName : getPropertyNames()) {
                rowMajorVariantProperties[flatIndex] = (float)variantPropertiesMap.get(propertyName)[rowIndex];
                ++flatIndex;
            }
        }
        return rowMajorVariantProperties;
    }

    private DMatrix getDMatrix(final int[] variantIndices) {
        final float[] propertiesArr = getRowMajorVariantProperties(variantIndices);
        final int[] perVariantOptimalMinGq = getPerVariantOptimalMinGq(variantIndices);
        if(propertiesArr.length != variantIndices.length * getNumProperties()) {
            throw new GATKException("rowMajorVariantProperties has length " + propertiesArr.length + ", should be " + variantIndices.length * getNumProperties());
        }
        return getDMatrix(propertiesArr, perVariantOptimalMinGq, variantIndices.length, getNumProperties());
    }

    private DMatrix getDMatrix(final double[] variantProperties) {
        final float[] arr = new float[variantProperties.length];
        final int[] perVariantOptimalMinGq = new int[1];
        for(int i = 0; i < variantProperties.length; ++i) {
            arr[i] = (float)variantProperties[i];
        }
        return getDMatrix(arr, perVariantOptimalMinGq, 1, variantProperties.length);
    }

    private DMatrix getDMatrix(final float[] propertiesArr, final int[] perVariantOptimalMinGq,
                               final int numRows, final int numColumns) {
        try {
            for(final float val : propertiesArr) {
                if(!Float.isFinite(val)) {
                    throw new GATKException("rowMajorVariantProperties contains a non-finite value (" + val + ")");
                }
            }
            final DMatrix dMatrix = new DMatrix(
                    propertiesArr, numRows, numColumns, Float.NaN
            );
            // Set baseline (initial prediction for min GQ)
            //final int baselineGq = getGenotypeQualitiesQuantile(initialMinGqQuantile);
            final int baselineGq = 0;
            final float baselinePredict = (float)baselineGq / (float)predictionScaleFactor;
            final float[] baseline = new float[numRows];
            final float[] weights = new float[numRows];
            final float[] labels = new float[numRows];
            for(int idx = 0; idx < numRows; ++idx) {
                baseline[idx] = baselinePredict;
                weights[idx] = (float) maxDiscoverableMendelianAc[idx];
                labels[idx] = (float) perVariantOptimalMinGq[idx] / (float)predictionScaleFactor;
            }
            dMatrix.setLabel(labels);
            dMatrix.setBaseMargin(baseline);
            dMatrix.setWeight(weights);
            return dMatrix;
        }
        catch(XGBoostError xgBoostError) {
            throw new GATKException("Error forming DMatrix", xgBoostError);
        }
    }

    private Map<String, Object> getXgboostParams() {
        return new HashMap<String, Object>() {
            private static final long serialVersionUID = 0L;
            {
                put("eta", eta);
                put("max_depth", maxDepth);
                put("gamma", gamma);
                put("subsample", subsample);
                put("colsample_bytree", colsampleByTree);
                put("colsample_bylevel", colsampleByLevel);
                put("min_child_weight", minChildWeight);
                put("validate_parameters", true);
                put("objective", "reg:squarederror");
                put("eval_metric", "rmse");
            }
        };
    }

    private float[] predictsToFloatMinGq(final float[][] predicts, float[] minGq) {
        if(minGq == null) {
            minGq = new float[predicts.length];
        }
        for(int idx = 0; idx < predicts.length; ++idx) {
            minGq[idx] = (float)predictionScaleFactor * predicts[idx][0];
        }
        return minGq;
    }

    private int[] predictsToIntMinGq(final float[][] predicts, int[] minGq) {
        if(minGq == null) {
            minGq = new int[predicts.length];
        }
        for(int idx = 0; idx < predicts.length; ++idx) {
            minGq[idx] = (int)ceil(predictionScaleFactor * predicts[idx][0]);
        }
        return minGq;
    }

    private class DataSubset {
        final DMatrix dMatrix;
        final int[] variantIndices;
        final List<Float> scores;
        final int[] intMinGq;
        final float[] floatMinGq;
        final float[] d1Loss;
        final float[] d2Loss;
        final double bestPossibleLoss;

        private int bestScoreInd;
        private float bestScore;

        DataSubset(final DMatrix dMatrix, final int[] indices, final boolean needDerivatives) {
            this.dMatrix = dMatrix;
            this.variantIndices = indices;
            scores = new ArrayList<>();
            bestScore = Float.POSITIVE_INFINITY;
            bestScoreInd = 0;
            intMinGq = new int[indices.length];
            floatMinGq = needDerivatives ? new float[indices.length] : null;
            d1Loss = needDerivatives ? new float[indices.length] : null;
            d2Loss = needDerivatives ? new float[indices.length] : null;
            bestPossibleLoss = getLoss(getPerVariantOptimalMinGq(indices), indices).toDouble();
            System.out.println("Best possible loss = " + bestPossibleLoss);
        }

        public int size() { return variantIndices.length; }

        public void calcScore(final float[][] rawPredictions) {
            predictsToIntMinGq(rawPredictions, intMinGq);
            final Loss loss = getLoss(intMinGq, variantIndices);
            appendScore(loss.toFloat());
            if(d1Loss != null) {
                // calculate derivatives
                predictsToFloatMinGq(rawPredictions, floatMinGq);
                calculateDerivatives(loss);
            }
        }

        void calculateDerivativesTargets(final Loss loss) {
            final int[] optimalMinGq = getPerVariantOptimalMinGq(variantIndices);
            final double squareFactor = predictionScaleFactor * predictionScaleFactor;
            final double deltaLoss = loss.toDouble() - bestPossibleLoss;
            if(deltaLoss <= 0) {
                throw new GATKException("Bad deltaLoss: " + deltaLoss);
            }
            System.out.println("deltaLoss=" + deltaLoss);
            for (int i = 0; i < variantIndices.length; ++i) {
                final double err = floatMinGq[i] - optimalMinGq[i];
                if(err == 0) {
                    d1Loss[i] = 0F;
                    d2Loss[i] = (float) (squareFactor * 2.0 / FastMath.max(deltaLoss, 1.0e-3));
                } else {
                    final double safeErr = FastMath.signum(err) * FastMath.max(FastMath.abs(err), deltaLoss);
                    d1Loss[i] = (float) (predictionScaleFactor * 2.0 * deltaLoss / safeErr);
                    d2Loss[i] = (float) (squareFactor * 2.0 * deltaLoss / (safeErr * safeErr));
                }
            }
        }

        void calculateDerivatives(final Loss loss) {
            final BinnedFilterSummaries backgroundFilterSummary
                    = getBackgroundFilterSummary(null, variantIndices, intMinGq);
            final double squareFactor = predictionScaleFactor * predictionScaleFactor;
            for (int i = 0; i < variantIndices.length; ++i) {
                final int variantIndex = variantIndices[i];
                final BinnedFilterSummaries filterSummary = getBinnedFilterSummary(intMinGq[i], variantIndex);
                final FilterQuality localOptimum = getOptimalVariantMinGq(
                    variantIndex, backgroundFilterSummary.subtract(filterSummary)
                );
                final double err = floatMinGq[i] - localOptimum.getMinGq();
                final double deltaLoss = loss.toDouble() - bestPossibleLoss;
                if(err == 0 || deltaLoss == 0) {
                    d1Loss[i] = 0F;
                    d2Loss[i] = (float) (squareFactor * 2.0 / FastMath.max(deltaLoss, 1.0e-3));
                } else {
                    final double safeErr = FastMath.signum(err) * FastMath.max(FastMath.abs(err), deltaLoss);
                    d1Loss[i] = (float) (predictionScaleFactor * 2.0 * deltaLoss / safeErr);
                    d2Loss[i] = (float) (squareFactor * 2.0 * deltaLoss / (safeErr * safeErr));
                }
            }
        }

        public float getLastScore() { return scores.get(scores.size() - 1); }

        public void appendScore(final float score) {
            scores.add(score);
            if(score < bestScore) {
                bestScore = score;
                bestScoreInd = getRound();
            }
        }

        public int getRound() {
            return scores.size() - 1;
        }

        public boolean isBestScore() {
            return bestScoreInd == getRound();
        }

        public boolean stop() {
            return getRound() == maxTrainingRounds || stopEarly();
        }

        public boolean stopEarly() {
            return earlyStoppingRounds > 0 && getRound() - bestScoreInd > earlyStoppingRounds;
        }
    }

    @Override
    protected boolean needsZScore() { return false; }

    @Override
    protected int predict(double[] variantProperties) {
        try {
            return Math.round(
                booster.predict(getDMatrix(variantProperties), true, 0)[0][0]
            );
        } catch(XGBoostError xgBoostError) {
            throw new GATKException("Error predicting", xgBoostError);
        }
    }

    @Override
    protected void loadModel(final InputStream inputStream) {
        try {
            booster = XGBoost.loadModel(inputStream);
        } catch(XGBoostError | IOException xgBoostError) {
            throw new GATKException("Error loading XGBoost model", xgBoostError);
        }
    }

    @Override
    protected void saveModel(final OutputStream outputStream) {
        try {
            booster.saveModel(outputStream);
        } catch(XGBoostError | IOException xgBoostError) {
            throw new GATKException("Error saving XGBoost model", xgBoostError);
        }
    }

    private Booster initializeBooster(final Map<String, DataSubset> dataSubsetMap) {
        final Map<String, DMatrix> watches = new HashMap<>();
        for(final Map.Entry<String, DataSubset> entry : dataSubsetMap.entrySet()) {
            if(!entry.getKey().equals(TRAIN_MAT_KEY)) {
                watches.put(entry.getKey(), entry.getValue().dMatrix);
            }
        }

        try {
            return XGBoost.train(dataSubsetMap.get(TRAIN_MAT_KEY).dMatrix, getXgboostParams(), 0, watches, null, null);
        } catch(XGBoostError xgBoostError) {
            throw new GATKException("Error creating Booster", xgBoostError);
        }
    }

    private boolean trainOneRound(final Booster booster, final Map<String, DataSubset> dataSubsets) {
        // evaluate booster on all data sets, calculate derivatives on training set
        final DataSubset validationData = dataSubsets.get(VALIDATION_MAT_KEY);
        if(printProgress > 0) {
            System.out.println("Evaluating round " + (validationData.getRound() + 1));
        }
        for(final Map.Entry<String, DataSubset> dataSubsetEntry : dataSubsets.entrySet()) {
            final String key = dataSubsetEntry.getKey();
            final DataSubset dataSubset = dataSubsetEntry.getValue();
            final float[][] predicts;
            try {
                predicts = booster.predict(dataSubset.dMatrix, true, 0);
            } catch(XGBoostError xgBoostError) {
                throw new GATKException("Error predicting " + key + " matrix, round " + dataSubset.getRound(), xgBoostError);
            }
            dataSubset.calcScore(predicts);
            if(key.equals(TRAIN_MAT_KEY)) {
                if(printProgress > 1) {
                    final int[] perVariantOptimalMinGq = getPerVariantOptimalMinGq(dataSubset.variantIndices);
                    System.out.println("predicts.size = [" + predicts.length + "," + predicts[0].length + "]");
                    System.out.println("minGq\toptimal\td1\td2");
                    IntStream.range(0, dataSubset.size())
                        .mapToObj()
                    for(int idx = 0; idx < 10; ++idx) {
                        System.out.println(dataSubset.intMinGq[idx] + "\t" + perVariantOptimalMinGq[idx]
                                           + "\t" + dataSubset.d1Loss[idx] + "\t" + dataSubset.d2Loss[idx]);
                    }
                }
            }
            if(printProgress > 0) {
                System.out.println("\t" + key + ": " + dataSubset.getLastScore());
            }
        }
        // check if booster needs to be saved, or if early stopping is necessary
        if(validationData.isBestScore()) {
            saveModelCheckpoint();
        }
        if(validationData.stop()) {
            return false;
        }
        // boost booster
        if(printProgress > 0) {
            System.out.println("Boosting round " + validationData.getRound());
        }
        try {
            final DataSubset trainData = dataSubsets.get(TRAIN_MAT_KEY);
            booster.boost(trainData.dMatrix, trainData.d1Loss, trainData.d2Loss);
        } catch(XGBoostError xgBoostError) {
            throw new GATKException("Error boosting round " + dataSubsets.get(TRAIN_MAT_KEY).getRound(), xgBoostError);
        }
        return true;
    }

    @Override
    protected void trainFilter() {
        final Map<String, DataSubset> dataSubsets = new HashMap<String, DataSubset>() {
            private static final long serialVersionUID = 0L;
            {
                put(TRAIN_MAT_KEY,
                    new DataSubset(getDMatrix(getTrainingIndices()), getTrainingIndices(), true));
                put(VALIDATION_MAT_KEY,
                    new DataSubset(getDMatrix(getValidationIndices()), getValidationIndices(), false));
            }
        };
        booster = initializeBooster(dataSubsets);
        //noinspection StatementWithEmptyBody
        while (trainOneRound(booster, dataSubsets))
            ;

        loadModelCheckpoint();
        final float[][] predicts;
        try {
            predicts = booster.predict(dataSubsets.get(TRAIN_MAT_KEY).dMatrix, true, 0);
        } catch(XGBoostError xgBoostError) {
            throw new GATKException("Error predicting final training minGq", xgBoostError);
        }
        final int[] minGq = predictsToIntMinGq(predicts, null);
        displayHistogram("Final training prediction histogram", Arrays.stream(minGq), true);
    }
}
