package org.broadinstitute.hellbender.tools.walkers.sv;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.ml.regression.RandomForestRegressionModel;
import org.apache.spark.ml.regression.RandomForestRegressor;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.broadinstitute.barclay.argparser.Argument;
import org.broadinstitute.barclay.argparser.CommandLineProgramProperties;
import org.broadinstitute.barclay.help.DocumentedFeature;
import org.broadinstitute.hellbender.cmdline.programgroups.StructuralVariantDiscoveryProgramGroup;

import java.io.InputStream;
import java.io.OutputStream;
import java.util.*;
import java.util.stream.Collectors;

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
public class MinGqVariantFilterRandomForest extends MinGqVariantFilterBase {
    @Argument(fullName="num-trees", shortName="nt", doc="Number of trees in random forest", optional=true, minValue=1)
    public int numTrees = 100;
    @Argument(fullName="max-depth", shortName="d", doc="Max depth of boosted decision tree", optional=true, minValue=1)
    public int maxDepth = 20;
    @Argument(fullName="max-feature-bins", shortName="mfb", doc="Max number of bins for continuous features", optional=true, minValue=1)
    public int maxFeatureBins = 1000;

    RandomForestRegressionModel forest;
    private final SparkSession sparkSession = SparkSession
            .builder()
            .appName("JavaRandomForestRegressorExample")
            .getOrCreate();;

    @Override
    public boolean needsZScore() { return false; }

    private Row getRow(final int variantIndex) {
        return RowFactory.create(
            getPropertyNames().stream()
                .map(propertyName -> variantPropertiesMap.get(propertyName))
                .toArray()
        );
    }

    private Dataset<Row> getDataFrame(final int[] variantIndices) {
        StructType schema = DataTypes.createStructType(
            getPropertyNames().stream()
                .map(propName -> DataTypes.createStructField(propName, DataTypes.DoubleType, false))
                .collect(Collectors.toList())
        );

        // need to set labels
        return sparkSession.createDataFrame(
            Arrays.stream(variantIndices).mapToObj(this::getRow).collect(Collectors.toList()),
            schema
        );
    }

    @Override
    public int predict(double[] variantProperties) {
        return (int)Math.round(forest.predict(new DenseVector(variantProperties)));
    }

    @Override
    protected void trainFilter() {
        final RandomForestRegressor regressor = new RandomForestRegressor()
            .setMaxDepth(maxDepth)
            .setMaxBins(maxFeatureBins)
            .setNumTrees(numTrees);
        forest = regressor.fit(getDataFrame(getTrainingIndices()));
    }

    @Override
    protected void saveModel(OutputStream outputStream) {
        throw new org.apache.commons.lang.NotImplementedException();
    }

    @Override
    protected void loadModel(InputStream inputStream) {
        throw new org.apache.commons.lang.NotImplementedException();
    }
}
