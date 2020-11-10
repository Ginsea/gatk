package org.broadinstitute.hellbender.tools.walkers.sv;

import ml.dmlc.xgboost4j.java.*;
import org.broadinstitute.barclay.argparser.Argument;
import org.broadinstitute.barclay.argparser.CommandLineProgramProperties;
import org.broadinstitute.barclay.help.DocumentedFeature;
import org.broadinstitute.hellbender.cmdline.programgroups.StructuralVariantDiscoveryProgramGroup;
import org.broadinstitute.hellbender.exceptions.GATKException;

import java.io.*;
import java.util.*;
import java.util.stream.IntStream;

import static org.apache.commons.math3.util.FastMath.ceil;

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
    @Argument(fullName="max-training-rounds", shortName="t", doc="Maximum number of rounds of training", optional=true, minValue=1)
    public int maxTrainingRounds = 100;
    @Argument(fullName="early-stopping-rounds", shortName="e", doc="Stop training if no improvement is made in validation set for this many rounds. Set <= 0 to disable.", optional=true)
    public int earlyStoppingRounds = 10;
    @Argument(fullName="initial-min-gq-quantile", shortName="q", doc="Initial guess for min GQ, as a quantile of gq in variants of trios.", optional=true)
    public double initialMinGqQuantile = 0.05;
    @Argument(fullName="learning-rate", shortName="r", doc="Learning rate for xgboost", optional=true)
    public double eta = 1.0;
    @Argument(fullName="max-depth", shortName="d", doc="Max depth of boosted decision tree", optional=true, minValue=1)
    public int maxDepth = 6;
    @Argument(fullName="gamma", doc="Regularization factor for xgboost", optional=true)
    public double gamma = 1.0e-9;
    @Argument(fullName="subsample", doc="Proportion of data selected for each tree", optional=true)
    public double subsample = 0.7;
    @Argument(fullName="colsample_by_tree", doc="Proportion of columns selected for each tree", optional=true)
    public double colsampleByTree = 0.7;
    @Argument(fullName="colsample_by_level", doc="Proportion of columns selected for each level of each tree", optional=true)
    public double colsampleByLevel = 1.0;

    private Booster booster = null;

    static final int DOWN_IDX = 0;
    static final int MIDDLE_IDX = 1;
    static final int UP_IDX = 2;
    private final int[] minGqForDerivs = new int[3];
    private final int[] numPassedForDerivs = new int[3];
    private final int[] numMendelianForDerivs = new int[3];
    private float[] d1NumPassed = null;
    private float[] d2NumPassed = null;
    private float[] d1NumMendelian = null;
    private float[] d2NumMendelian = null;

    private static final String TRAIN_MAT_KEY = "train";
    private static final String VALIDATION_MAT_KEY = "validation";


    private int getGenotypeQualitiesQuantile(final double quantile) {
        if(quantile < 0 || quantile > 1) {
            throw new GATKException("quantile should be in the closed range [0.0, 1.0]");
        }
        final int numGenotypeQualities = getNumVariants() * getNumTrios() * 3;
        final int quantileInd = (int)Math.round(quantile * numGenotypeQualities);
        return genotypeQualitiesTensor.stream().flatMapToInt(
                mat -> Arrays.stream(mat).flatMapToInt(Arrays::stream)
        ).sorted().skip(quantileInd).findFirst().orElseThrow(
                () -> new GATKException("Unable to obtain " + quantile + " of allele counts. This is a bug.")
        );
    }

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

    private DMatrix getDMatrix(final int[] rowIndices) {
        final float[] arr = getRowMajorVariantProperties(rowIndices);
        if(arr.length != rowIndices.length * getNumProperties()) {
            throw new GATKException("rowMajorVariantProperties has length " + arr.length + ", should be " + rowIndices.length * getNumProperties());
        }
        return getDMatrix(arr, rowIndices.length, getNumProperties());
    }

    private DMatrix getDMatrix(final double[] variantProperties) {
        final float[] arr = new float[variantProperties.length];
        for(int i = 0; i < variantProperties.length; ++i) {
            arr[i] = (float)variantProperties[i];
        }
        return getDMatrix(arr, 1, variantProperties.length);
    }

    private DMatrix getDMatrix(final float[] arr, final int numRows, final int numColumns) {
        try {
            for(final float val : arr) {
                if(!Float.isFinite(val)) {
                    throw new GATKException("rowMajorVariantProperties contains a non-finite value (" + val + ")");
                }
            }
            final DMatrix dMatrix = new DMatrix(
                    arr, numRows, numColumns, Float.NaN
            );
            // Add dummy labels so that XGBoost thinks there are two classes
            // Set baseline (initial prediction for min GQ)
            final int baselineGq = getGenotypeQualitiesQuantile(initialMinGqQuantile);
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
                put("validate_parameters", true);
                put("objective", "reg:squarederror");
                put("eval_metric", "rmse");
            }
        };
    }

    private float[] predictsToMinGq(final float[][] predicts) {
        final float[] minGq = new float[predicts.length];
        for(int idx = 0; idx < predicts.length; ++idx) {
            minGq[idx] = (float)predictionScaleFactor * predicts[idx][0];
        }
        return minGq;
    }

    private int[] predictsToIntMinGq(final float[][] predicts) {
        final int[] minGq = new int[predicts.length];
        for(int idx = 0; idx < predicts.length; ++idx) {
            minGq[idx] = (int)Math.ceil(predictionScaleFactor * predicts[idx][0]);
        }
        return minGq;
    }

    private class DataSubset {
        final DMatrix dMatrix;
        final int[] indices;
        final List<Float> scores;

        private int bestScoreInd;
        private float bestScore;

        DataSubset(final DMatrix dMatrix, final int[] indices) {
            this.dMatrix = dMatrix;
            this.indices = indices;
            this.scores = new ArrayList<>();
            this.bestScore = Float.POSITIVE_INFINITY;
            this.bestScoreInd = 0;
        }

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

    private void bracketFiniteDifferences(final TrioFilterSummary predictSummary,
                                          final int[][] alleleCounts, final int[][] genotypeQualities) {
        final int[] candidateMinGqs = getCandidateMinGqs(alleleCounts, genotypeQualities, predictSummary.minGq).toArray();
        final int lastIdx = candidateMinGqs.length - 1;
        final TrioFilterSummary downSummary, middleSummary, upSummary;
        if(candidateMinGqs.length == 0) {
            downSummary = predictSummary.shiftMinGq(-1); // set downSummary minGq to 1 less than predictMinGq
            middleSummary = predictSummary;
            upSummary = predictSummary.shiftMinGq(+1);
        } else if(candidateMinGqs[0] > predictSummary.minGq) {
            // no candidates available less than predictMinGq
            if(candidateMinGqs.length == 1) {
                downSummary = predictSummary.shiftMinGq(-1); // set downSummary minGq to 1 less than predictMinGq
                middleSummary = predictSummary;
                upSummary = getTrioFilterSummary(candidateMinGqs[0], alleleCounts, genotypeQualities);
            } else {
                downSummary = predictSummary;
                middleSummary = getTrioFilterSummary(candidateMinGqs[0], alleleCounts, genotypeQualities);
                upSummary = getTrioFilterSummary(candidateMinGqs[1], alleleCounts, genotypeQualities);
            }
        } else if(candidateMinGqs[lastIdx] < predictSummary.minGq) {
            // no candidates available greater than predictMinGq
            if(candidateMinGqs.length == 1) {
                downSummary = getTrioFilterSummary(candidateMinGqs[lastIdx], alleleCounts, genotypeQualities);
                middleSummary = predictSummary;
                upSummary = predictSummary.shiftMinGq(+1);
            } else {
                downSummary = getTrioFilterSummary(candidateMinGqs[lastIdx - 1], alleleCounts, genotypeQualities);
                middleSummary = getTrioFilterSummary(candidateMinGqs[lastIdx], alleleCounts, genotypeQualities);
                upSummary = predictSummary;
            }
        } else {
            // There is a candidate > predictMinGq and a candidate < predictMinGq, use those
            final int upIdx = IntStream.range(0, candidateMinGqs.length)
                    .filter(i -> candidateMinGqs[i] > predictSummary.minGq).findFirst().orElseThrow(
                            () -> {throw new GATKException("Error bracketing derivatives, this is a bug.");}
                    );
            if(upIdx < 1) {
                throw new GATKException("Error bracketing derivatives, this is a bug");
            }
            downSummary = getTrioFilterSummary(candidateMinGqs[upIdx - 1], alleleCounts, genotypeQualities);
            middleSummary = predictSummary;
            upSummary = getTrioFilterSummary(candidateMinGqs[upIdx], alleleCounts, genotypeQualities);
        }
        minGqForDerivs[DOWN_IDX] = downSummary.minGq;
        minGqForDerivs[MIDDLE_IDX] = middleSummary.minGq;
        minGqForDerivs[UP_IDX] = upSummary.minGq;

        numPassedForDerivs[DOWN_IDX] = (int)downSummary.numPassed;
        numPassedForDerivs[MIDDLE_IDX] = (int)middleSummary.numPassed;
        numPassedForDerivs[UP_IDX] = (int)upSummary.numPassed;

        numMendelianForDerivs[DOWN_IDX] = (int)downSummary.numMendelian;
        numMendelianForDerivs[MIDDLE_IDX] = (int)middleSummary.numMendelian;
        numMendelianForDerivs[UP_IDX] = (int)upSummary.numMendelian;
    }

    private float sqr(final float x) { return x * x; }


    private void setDerivsFromDifferences(final float xTrue, final int[] x, final int[] y,
                                          final int derivativeIdx, final float[] d1, final float[] d2) {
        // model as quadratic: y = a * x^2 + b * x + c
        final float denom = (float)((x[2] - x[1]) * (x[2] - x[0]) * (x[1] - x[0]));
        final float a = ((x[2] - x[1]) * (y[1] - y[0]) - (x[1] - x[0]) * (y[2] - y[1])) / denom;
        final float b = ((sqr(x[2]) - sqr(x[1])) * (y[1] - y[0]) - (sqr(x[1]) - sqr(x[0])) * (y[2] - y[1])) / denom;
        final float d2_x = 2f * a;
        d1[derivativeIdx] = (float)(predictionScaleFactor) * (d2_x * xTrue + b);
        if(d2 != null) {
            d2[derivativeIdx] = (float)(predictionScaleFactor * predictionScaleFactor) * d2_x;
        }
    }

    protected float getLoss(final float[] minGq, final int[] variantIndices) {
        if(minGq.length != variantIndices.length) {
            throw new GATKException(
                    "Length of minGq (" + minGq.length + ") does not match length of variantIndices (" + variantIndices.length + ")"
            );
        }
        long numPassed = 0;
        long numMendelian = 0;
        long numDiscoverable = 0;
        for(int idx = 0; idx < minGq.length; ++idx) {
            final int variantIndex = variantIndices[idx];
            final TrioFilterSummary variantFilterSummary = getTrioFilterSummary((int)ceil(minGq[idx]), variantIndex);
            // update number of non-REF alleles, number of non-REF that pass filter, and number that are in a trio
            // compatible with Mendelian inheritance
            numPassed += variantFilterSummary.numPassed;
            numMendelian += variantFilterSummary.numMendelian;
            numDiscoverable += maxDiscoverableMendelianAc[variantIndex];
        }
        // report loss = 1.0 - f1 (algorithms expect to go downhill)
        return 1F - (float)getF1(numDiscoverable, numMendelian, numPassed);
    }

    protected List<float[]> getLossDerivs(final float[] minGq, final int[] variantIndices, final int derivativeOrder) {
        if(minGq.length != variantIndices.length) {
            throw new GATKException(
                    "Length of minGq (" + minGq.length + ") does not match length of variantIndices (" + variantIndices.length + ")"
            );
        }
        if(derivativeOrder <= 0 || derivativeOrder >= 3) {
            throw new GATKException("derivativeOrder must be 1 or 2. Supplied value is " + derivativeOrder);
        }
        final float[] d1Loss = new float[minGq.length];
        final float[] d2Loss = derivativeOrder > 1 ? new float[minGq.length] : null;
        if(d1NumPassed == null) {
            d1NumPassed = new float[getNumVariants()];
            d1NumMendelian = new float[getNumVariants()];
        }
        if(derivativeOrder > 1) {
            if(d2NumPassed == null) {
                d2NumPassed = new float[getNumVariants()];
                d2NumMendelian = new float[getNumVariants()];
            }
        }

        long numPassed = 0;
        long numMendelian = 0;
        for(int idx = 0; idx < minGq.length; ++idx) {
            final boolean debug = (printProgress > 1) && (idx < 5);

            final float variantMinGq = minGq[idx];
            final int variantIdx = variantIndices[idx];

            final int[][] variantAlleleCounts = alleleCountsTensor.get(variantIdx);
            final int[][] variantGenotypeQualities = genotypeQualitiesTensor.get(variantIdx);

            final int predictMinGq = (int)ceil(variantMinGq);
            // update number of non-REF alleles, number of non-REF that pass filter, and number that are in a trio
            // compatible with Mendelian inheritance
            TrioFilterSummary predictSummary = getTrioFilterSummary(predictMinGq, variantAlleleCounts, variantGenotypeQualities);
            numPassed += predictSummary.numPassed;
            numMendelian += predictSummary.numMendelian;
            bracketFiniteDifferences(predictSummary, variantAlleleCounts, variantGenotypeQualities);
            setDerivsFromDifferences(variantMinGq, minGqForDerivs, numPassedForDerivs, idx,
                    d1NumPassed, derivativeOrder > 1 ? d2NumPassed : null);
            setDerivsFromDifferences(variantMinGq, minGqForDerivs, numMendelianForDerivs, idx,
                    d1NumMendelian, derivativeOrder > 1 ? d2NumMendelian : null);

            if(debug) {
                System.out.println("minGq: " + Arrays.toString(minGqForDerivs));
                System.out.println("numPassed: " + Arrays.toString(numPassedForDerivs));
                System.out.println("d1,d2NumPassed: " + d1NumPassed[idx] + ", " + d2NumPassed[idx]);
                System.out.println("numMendelian: " + Arrays.toString(numMendelianForDerivs));
                System.out.println("d1,d2NumMendelian: " + d1NumMendelian[idx] + ", " + d2NumMendelian[idx]);
            }
        }

        // calculate f1 score:
        //     f1 = 2.0 / (1.0 / recall + 1.0 / precision)
        //     recall = numMendelian / numNonRef
        //     precision = numMendelian / numPassed
        //     -> f1 = 2.0 * numMendelian / (numNonRef + numPassed)
        final float denom = (float)(numDiscoverableMendelianAc + numPassed);
        final float f1 = 2f * numMendelian / denom;
        // loss is 1.0 - f1  (algorithms expect to go downhill)
        final float loss = 1f - f1;
        for(int idx = 0; idx < minGq.length; ++idx) {
            d1Loss[idx] = (f1 * d1NumPassed[idx] - 2f * d1NumMendelian[idx]) / denom;
            if(derivativeOrder > 1) {
                d2Loss[idx] = (f1 * d2NumPassed[idx] - 2f * d2NumMendelian[idx] - 2f * d1NumPassed[idx] * d1Loss[idx]) / denom;
            }
        }
        final float[] lossArr = new float[] {loss};
        return derivativeOrder > 1 ? Arrays.asList(lossArr, d1Loss, d2Loss) : Arrays.asList(lossArr, d1Loss);
    }

    private boolean trainOneRound(final Booster booster, final Map<String, DataSubset> dataSubsets) {
        // evaluate booster on all data sets, calculate derivatives on training set
        final DataSubset validationData = dataSubsets.get(VALIDATION_MAT_KEY);
        if(printProgress > 0) {
            System.out.println("Evaluating round " + (validationData.getRound() + 1));
        }
        List<float[]> derivs = null;
        for(final Map.Entry<String, DataSubset> dataSubsetEntry : dataSubsets.entrySet()) {
            final String key = dataSubsetEntry.getKey();
            final DataSubset dataSubset = dataSubsetEntry.getValue();
            final float[][] predicts;
            try {
                predicts = booster.predict(dataSubset.dMatrix, true, 0);
            } catch(XGBoostError xgBoostError) {
                throw new GATKException("Error predicting " + key + " matrix, round " + dataSubset.getRound(), xgBoostError);
            }
            final float[] minGq = predictsToMinGq(predicts);
            final float score;
            if(key.equals(TRAIN_MAT_KEY)) {
                derivs = getLossDerivs(minGq, dataSubset.indices, 2);
                score = derivs.get(0)[0];
                if(printProgress > 1) {
                    System.out.println("predicts.size = [" + predicts.length + "," + predicts[0].length + "]");
                    System.out.println("minGq\td1\td2");
                    for(int idx = 0; idx < 10; ++idx) {
                        System.out.println(minGq[idx] + "\t" + derivs.get(1)[idx] + "\t" + derivs.get(2)[idx]);
                    }
                }
            } else {
                score = getLoss(minGq, dataSubset.indices);
            }
            dataSubset.appendScore(score);
            if(printProgress > 0) {
                System.out.println("\t" + key + ": " + score);
            }
        }
        if(derivs == null) {
            throw new GATKException("derivs was not assigned a value. This is a bug.");
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
            booster.boost(dataSubsets.get(TRAIN_MAT_KEY).dMatrix, derivs.get(1), derivs.get(2));
        } catch(XGBoostError xgBoostError) {
            throw new GATKException("Error boosting round " + dataSubsets.get(TRAIN_MAT_KEY).getRound(), xgBoostError);
        }
        return true;
    }

    private void initializeHelperArrays() {
        d1NumPassed = new float[getNumVariants()];
        d1NumMendelian = new float[getNumVariants()];
        d2NumPassed = new float[getNumVariants()];
        d2NumMendelian = new float[getNumVariants()];
    }

    @Override
    protected void trainFilter() {
        initializeHelperArrays();

        final Map<String, DataSubset> dataSubsets = new HashMap<String, DataSubset>() {
            private static final long serialVersionUID = 0L;

            {
                put(TRAIN_MAT_KEY, new DataSubset(getDMatrix(getTrainingIndices()), getTrainingIndices()));
                put(VALIDATION_MAT_KEY, new DataSubset(getDMatrix(getValidationIndices()), getValidationIndices()));
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
        final int[] minGq = predictsToIntMinGq(predicts);
        displayGqHistogram("Final training prediction histogram", Arrays.stream(minGq), true);

    }
}
