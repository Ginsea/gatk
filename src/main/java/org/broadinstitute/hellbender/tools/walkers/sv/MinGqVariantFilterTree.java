package org.broadinstitute.hellbender.tools.walkers.sv;

import net.minidev.json.JSONObject;
import net.minidev.json.JSONValue;
import org.apache.commons.math3.util.FastMath;
import org.broadinstitute.barclay.argparser.Argument;
import org.broadinstitute.barclay.argparser.CommandLineProgramProperties;
import org.broadinstitute.barclay.help.DocumentedFeature;
import org.broadinstitute.hellbender.cmdline.programgroups.StructuralVariantDiscoveryProgramGroup;
import org.broadinstitute.hellbender.exceptions.GATKException;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

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
public class MinGqVariantFilterTree extends MinGqVariantFilterBase {
    @Argument(fullName="max-depth", shortName="d", doc="Max depth of boosted decision tree", optional=true, minValue=1)
    public int maxDepth = 20;
    @Argument(fullName="early-stopping-rounds", shortName="e", doc="Stop training if no improvement is made in testing set for this many rounds. Set <= 0 to disable.", optional=true)
    public int earlyStoppingRounds = 10;
    @Argument(fullName="num-check-candidates", shortName="c", doc="Number of candidate splits to evaluate before choosing next split for tree", optional=true, minValue=1)
    public int numCheckCandidates = 5;

    DecisionNode rootNode = null;

    class IndexSplit {
        final List<Integer> indicesLow;
        final List<Integer> indicesHigh;

        IndexSplit(final List<Integer> indices, final String propertyName, final double splitValue) {
            this(indices, variantPropertiesMap.get(propertyName), splitValue);
        }

        IndexSplit(final List<Integer> indices, final double[] propertyValues, final double splitValue) {
            this.indicesLow = new ArrayList<>(indices.size());
            this.indicesHigh = new ArrayList<>(indices.size());
            for (final int index : indices) {
                if (propertyValues[index] >= splitValue) {
                    indicesHigh.add(index);
                } else {
                    indicesLow.add(index);
                }
            }
        }

        int[] getLow() { return indicesLow.stream().mapToInt(Integer::intValue).toArray(); }
        int[] getHigh() { return indicesHigh.stream().mapToInt(Integer::intValue).toArray(); }
    }

    class DecisionNode {
        private int minGq;
        private final DecisionNode parent;
        private DecisionNode childLow;
        private DecisionNode childHigh;
        private String splitColumn;
        private double splitColumnValue;
        private transient List<CandidateSplit> candidateSplits;

        DecisionNode(final JSONObject jsonObject, final DecisionNode parent) {
            this.minGq = (Integer)jsonObject.get("minGq");
            this.parent = parent;
            this.splitColumn = (String)jsonObject.getOrDefault("splitColumn", null);
            this.splitColumnValue = this.splitColumn == null ?
                Double.NaN :
                getDoubleFromJSON(jsonObject.get("splitColumnValue"));
            this.childLow = jsonObject.containsKey("childLow") ?
                new DecisionNode((JSONObject)jsonObject.get("childLow"), this) :
                null;
            this.childHigh = jsonObject.containsKey("childHigh") ?
                new DecisionNode((JSONObject)jsonObject.get("childHigh"), this) :
                null;
        }

        DecisionNode(final int minGq, final DecisionNode parent) {
            this(minGq, parent, null, null, null, Double.NaN);
        }

        DecisionNode(final int minGq, final DecisionNode parent, final DecisionNode childLow,
                     final DecisionNode childHigh, final String splitColumn, final double splitColumnValue) {
            this.minGq = minGq;
            this.parent = parent;
            this.childLow = childLow;
            this.childHigh = childHigh;
            this.splitColumn = splitColumn;
            this.splitColumnValue = splitColumnValue;
            this.candidateSplits = null;
        }

        JSONObject toJson() {
            final JSONObject jsonObject = new JSONObject();
            jsonObject.put("minGq", minGq);
            if(splitColumn != null) {
                jsonObject.put("splitColumn", splitColumn);
                jsonObject.put("splitColumnValue", splitColumnValue);
            }
            if(childLow != null) {
                jsonObject.put("childLow", childLow.toJson());
            }
            if(childHigh != null) {
                jsonObject.put("childHigh", childHigh.toJson());
            }
            return jsonObject;
        }

        final int getNumNodes() { return isLeaf() ? 1 : 1 + childLow.getNumNodes() + childHigh.getNumNodes(); }

        final int getMaxDepth() { return isLeaf() ? 1 : 1 + FastMath.max(childLow.getMaxDepth(), childHigh.getMaxDepth()); }

        final int getDepth() {
            return isRoot() ? 1 : 1 + parent.getDepth();
        }

        boolean isRoot() {
            return parent == null;
        }

        boolean isLeaf() {
            return (childLow == null) && (childHigh == null);
        }

        DecisionNode getRoot() { return isRoot() ? this : parent.getRoot(); }

        double getLoss(final int[] variantIndices) {
            return variantIndices.length > 0 ?
                MinGqVariantFilterTree.this.getLoss(predict(variantIndices), variantIndices) :
                Double.NaN;
        }

        int[] predict(final int[] variantIndices) {
            return predict(getVariantProperties(variantIndices));
        }

        int predict(final double[] properties) {
            if(isLeaf()) {
                return minGq;
            } else {
                final int columnIndex = MinGqVariantFilterTree.this.getPropertyNames().indexOf(splitColumn);
                return properties[columnIndex] >= splitColumnValue ?
                    childHigh.predict(properties) :
                    childLow.predict(properties);
            }
        }

        int[] predict(final Map<String, double[]> propertiesMap) {
            final int numRows = propertiesMap.values().stream().findFirst().orElseThrow(
                    () -> new GATKException("Properties have no columns")
            ).length;
            final int[] predictValues = new int [numRows];
            final List<Integer> predictRows = IntStream.range(0, numRows).boxed().collect(Collectors.toList());
            predictInto(propertiesMap, predictValues, predictRows);
            return predictValues;
        }

        private void predictInto(final Map<String, double[]> propertiesMap,
                                 final int[] predictValues, final List<Integer> predictRows) {
            if(predictRows.isEmpty()) {
                return;
            }
            if(isLeaf()) {
                if(printProgress > 5) {
                    System.out.println("\t\t\tnode_" + System.identityHashCode(this) + ": minGq=" + minGq + " (" + predictRows.size() + " rows)");
                }
                for(final int row : predictRows) {
                    predictValues[row] = minGq;
                }
            } else {
                final IndexSplit childIndexSplit =
                    new IndexSplit(predictRows, propertiesMap.get(splitColumn), splitColumnValue);
                childLow.predictInto(propertiesMap, predictValues, childIndexSplit.indicesLow);
                childHigh.predictInto(propertiesMap, predictValues, childIndexSplit.indicesHigh);
            }
        }

        final List<Integer> getTrainingIndices() {
            if(isRoot()) {
                return Arrays.stream(MinGqVariantFilterTree.this.getTrainingIndices()).boxed().collect(Collectors.toList());
            } else {
                final IndexSplit parentIndexSplit = new IndexSplit(
                        parent.getTrainingIndices(), parent.splitColumn, parent.splitColumnValue
                );
                return parent.childLow == this ? parentIndexSplit.indicesLow : parentIndexSplit.indicesHigh;
            }
        }

        List<CandidateSplit> getCandidateSplits() {
            if(getDepth() >= maxDepth) {
                return new ArrayList<>();
            } else if(isLeaf()) {
                if(candidateSplits == null) {
                    final int[] trainingIndices = getTrainingIndices().stream().mapToInt(Integer::intValue).toArray();
                    final Map<String, double[]> trainingData = getVariantProperties(trainingIndices);
                    final int[] optimalMinGq = getPerVariantOptimalMinGq(trainingIndices);
                    candidateSplits = trainingData.entrySet().stream().map(
                        entry -> new CandidateSplit(this, entry.getKey(), entry.getValue(), optimalMinGq)
                    )
                    .filter(candidateSplit -> candidateSplit.gain > 0)
                    .collect(Collectors.toList());
                }
                return candidateSplits;
            } else {
                return Stream.of(childLow.getCandidateSplits(), childHigh.getCandidateSplits())
                        .flatMap(Collection::stream)
                        .collect(Collectors.toList());
            }
        }

        void removeCandidateSplit(final CandidateSplit candidateSplit) {
            candidateSplits.remove(candidateSplit);
        }

        void implementSplit(final CandidateSplit candidateSplit) {
            childLow = new DecisionNode(candidateSplit.minGqLow, this);
            childHigh = new DecisionNode(candidateSplit.minGqHigh, this);
            splitColumn = candidateSplit.column;
            splitColumnValue = candidateSplit.splitValue;
        }

        void undoSplit() {
            childLow = null;
            childHigh = null;
            splitColumn = null;
            splitColumnValue = Double.NaN;
        }
    }

    class CandidateSplit {
        final DecisionNode node;
        final String column;
        final double splitValue;
        final double gain;
        int minGqLow;
        int minGqHigh;
        double trainingLoss;
        double validationLoss;

        CandidateSplit(final DecisionNode node, final String column, final double[] columnValues,
                       final int[] optimalMinGq) {
            this.node = node;
            this.column = column;
            this.trainingLoss = Double.NaN;
            this.validationLoss = Double.NaN;
            this.minGqLow = node.minGq;
            this.minGqHigh = node.minGq;

            final double[] candidateSplitValues = Arrays.stream(columnValues).sorted().distinct().toArray();
            if(candidateSplitValues.length <= 1) {
                gain = Double.NEGATIVE_INFINITY;
                splitValue = Double.NaN;
            } else {
                final double nodeError = getCandidateSplitError(optimalMinGq, columnValues, Double.NEGATIVE_INFINITY);
                int bestSplitIndex = -1;
                double bestSplitError = Double.POSITIVE_INFINITY;
                // start from index 1 so as to not consider grouping all values together
                for(int splitIndex = 1; splitIndex < candidateSplitValues.length; ++splitIndex) {
                    final double splitError = getCandidateSplitError(
                        optimalMinGq, columnValues, candidateSplitValues[splitIndex]
                    );
                    if(bestSplitIndex < 0 || splitError < bestSplitError) {
                        bestSplitIndex = splitIndex;
                        bestSplitError = splitError;
                    }
                }
                splitValue = candidateSplitValues[bestSplitIndex];
                gain = nodeError - bestSplitError;
            }
        }

        void delete() { node.removeCandidateSplit(this); }

        void addToTree() { node.implementSplit(this); }

        void optimize(final int[] trainingIndices, final int[] validationIndices) {
            final IndexSplit trainingIndexSplit = new IndexSplit(node.getTrainingIndices(), column, splitValue);
            if(printProgress > 4) {
                System.out.println("\t\t\tSplit column=" + column + ", value=" + splitValue + " (gain=" + gain + ")");
                System.out.println("\t\t\t" + trainingIndexSplit.indicesLow.size() + " low inds, "
                                   + trainingIndexSplit.indicesHigh.size() + " high inds");
            }

            // Split the node, but make the split have no effect by setting child minGq equal to parent
            this.minGqLow = node.minGq;
            this.minGqHigh = node.minGq;
            this.addToTree();

            final DecisionNode rootNode = node.getRoot();
            FilterQuality trainFilterQuality;
            double lastLoss = Double.POSITIVE_INFINITY;
            int iteration = 0;
            while(true) {
                ++iteration;
                // Iteratively optimize the minGq of the children
                int[] minGq = rootNode.predict(trainingIndices);
                trainFilterQuality = getOptimalMinGq(trainingIndexSplit.getLow(), trainingIndices, minGq);
                node.childLow.minGq = trainFilterQuality.minGq;
                if(printProgress > 4) {
                    System.out.println("\t\t\t" + iteration + ": low split, minGq=" + trainFilterQuality.minGq + ", loss=" + trainFilterQuality.loss);
                }
                if(trainFilterQuality.loss >= lastLoss) {
                    break;
                }
                lastLoss = trainFilterQuality.loss;

                minGq = rootNode.predict(trainingIndices);
                trainFilterQuality = getOptimalMinGq(trainingIndexSplit.getHigh(), trainingIndices, minGq);
                if(printProgress > 4) {
                    System.out.println("\t\t\t" + iteration + ": high split, minGq=" + trainFilterQuality.minGq + ",loss=" + trainFilterQuality.loss);
                }
                node.childHigh.minGq = trainFilterQuality.minGq;
                if(trainFilterQuality.loss >= lastLoss) {
                    break;
                }
                lastLoss = trainFilterQuality.loss;
            }

            // finished iterating, arrived at stable minimum
            trainingLoss = trainFilterQuality.loss;
            validationLoss = rootNode.getLoss(validationIndices);
            minGqLow = node.childLow.minGq;
            minGqHigh = node.childHigh.minGq;
            if(printProgress > 4) {
                System.out.println("\t\t\tfinal: minGq low=" + minGqLow + ", high=" + minGqHigh);
                System.out.println("\t\t\t       training loss=" + trainingLoss + ", validation loss=" + validationLoss);
            }

            // clean-up
            node.undoSplit();
        }

        /** Model node error as sum of square errors from constant prediction vs optimal min GQ. Note, this doesn't
         * reflect actual change in f1, so this is an approximation used to cheaply select the best candidate split
         */
        private double getCandidateSplitError(final int[] optimalMinGq, final double[] columnValues,
                                              final double candidateSplitValue) {
            long lowSum = 0;  long lowSqrSum = 0; int numLow = 0;
            long highSum = 0; long highSqrSum = 0; int numHigh = 0;
            for(int index = 0; index < columnValues.length; ++index) {
                final double columnValue = columnValues[index];
                final int minGq = optimalMinGq[index];
                if(columnValue >= candidateSplitValue) {
                    highSum += minGq;
                    highSqrSum += minGq * minGq;
                    ++numHigh;
                } else {
                    lowSum += minGq;
                    lowSqrSum += minGq * minGq;
                    ++numLow;
                }
            }
            final double sumSquareErrorsLow = numLow == 0 ? 0.0 :
                (lowSqrSum * numLow - lowSum * lowSum) / (double) numLow;
            final double sumSquareErrorsHigh = numHigh == 0 ? 0.0 :
                (highSqrSum * numHigh - highSum * highSum) / (double) numHigh;
            return sumSquareErrorsLow + sumSquareErrorsHigh;
        }
    }

    @Override
    protected boolean needsZScore() { return false; }

    @Override
    protected int predict(final double[] variantProperties) { return rootNode.predict(variantProperties); }

    @Override
    protected void saveModel(final OutputStream outputStream) {
        try {
            outputStream.write(rootNode.toJson().toJSONString().getBytes());
        } catch(IOException ioException) {
            throw new GATKException("Error saving data summary json", ioException);
        }
    }

    @Override
    protected void loadModel(final InputStream inputStream) {
        rootNode = new DecisionNode((JSONObject) JSONValue.parse(inputStream), null);
    }

    private void printTreeSummary(final double trainingLoss, final double validationLoss, final double testingLoss) {
        System.out.println("Training. " + rootNode.getNumNodes() + " nodes, max depth = " + rootNode.getMaxDepth());
        System.out.println("\ttraining loss=" + trainingLoss + ", validation loss=" + validationLoss + ", testing loss=" + testingLoss);
    }

    @Override
    protected void trainFilter() {
        final int[] trainingIndices = getTrainingIndices();
        final int[] validationIndices = getValidationIndices();
        final int[] testingIndices = getTestingIndices();

        double trainingLoss;
        if(rootNode == null) {
            FilterQuality treeFilterQuality = getOptimalMinGq(trainingIndices, trainingIndices, null);
            rootNode = new DecisionNode(treeFilterQuality.minGq, null);
            trainingLoss = treeFilterQuality.loss;
            if(printProgress > 0) {
                System.out.println("Found root minGQ=" + rootNode.minGq);
            }
        } else {
            if(printProgress > 0) {
                System.out.println("Resuming from previously saved tree.");
            }
            trainingLoss = rootNode.getLoss(trainingIndices);
        }
        double validationLoss = rootNode.getLoss(validationIndices);
        double testingLoss = rootNode.getLoss(testingIndices);
        double bestTestingLoss = testingLoss;
        int roundsSinceBestModel = 0;

        while(true) {
            if(printProgress > 0) {
                printTreeSummary(trainingLoss, validationLoss, testingLoss);
            }
            // Get potentialSplits from current tree, sorted with largest potential gain first
            final List<CandidateSplit> potentialSplits = rootNode.getCandidateSplits()
                    .stream().sorted(Comparator.comparingDouble(a -> -a.gain))
                    .collect(Collectors.toList());
            if(printProgress > 1) {
                System.out.println("\tFound " + potentialSplits.size() + " potential splits.");
            }
            CandidateSplit selectedSplit = null;
            double bestLoss = trainingLoss;
            for(int checkCandidate = 0; checkCandidate < FastMath.min(numCheckCandidates, potentialSplits.size()); ++checkCandidate) {
                if(printProgress > 2) {
                    System.out.println("\t\t optimizing split " + (checkCandidate + 1) + "/" + FastMath.min(numCheckCandidates, potentialSplits.size()));
                }
                final CandidateSplit checkSplit = potentialSplits.get(checkCandidate);
                checkSplit.optimize(trainingIndices, validationIndices);
                if(checkSplit.trainingLoss > trainingLoss || checkSplit.validationLoss > validationLoss) {
                    // This split doesn't improve at all, or validate. Never consider it again.
                    // Note: allow exact equality, which can happen if the splits both have the same minGQ as the parent
                    //       which may happen if the data needs to be split more than once before improvment can be
                    //       obtained
                    checkSplit.delete();
                } else if(checkSplit.trainingLoss <= bestLoss) {
                    // This is the current front-runner
                    selectedSplit = checkSplit;
                    bestLoss = checkSplit.trainingLoss;
                }
            }
            if(selectedSplit == null) {
                // Was not able to find a useful split. Stop fitting tree
                if(printProgress > 0) {
                    System.out.println("Stopping training due to no eligible/successful splits for decision tree.");
                }
                break;
            } else {
                // Take this split.
                // 1) add split to tree
                if(printProgress > 2) {
                    System.out.println("\t\tSelected split " + potentialSplits.indexOf(selectedSplit));
                }
                selectedSplit.addToTree();
                // 2) set current best training and validation losses
                trainingLoss = selectedSplit.trainingLoss;
                validationLoss = selectedSplit.validationLoss;
                testingLoss = rootNode.getLoss(testingIndices);
                if(testingLoss > bestTestingLoss) {
                    // model improvements aren't cross-validating on testing set
                    ++roundsSinceBestModel;
                    if(earlyStoppingRounds > 0 && roundsSinceBestModel > earlyStoppingRounds) {
                        if(printProgress > 0) {
                            System.out.println("Stopping training early due to lack of improvement on the testing set.");
                        }
                        break;
                    }
                } else {
                    // save checkpoint to best-performing tree
                    saveModelCheckpoint();
                    roundsSinceBestModel = 0;
                    bestTestingLoss = testingLoss;
                }
            }
        }

        if(roundsSinceBestModel == 0) {
            if(printProgress > 0) {
                System.out.println("Keeping current tree, as it is the best tree on the testing set.");
            }
        } else {
            if(printProgress > 0) {
                System.out.println("Restoring tree with best performance on the testing set.");
            }
            loadModelCheckpoint();
            trainingLoss = rootNode.getLoss(trainingIndices);
            validationLoss = rootNode.getLoss(validationIndices);
            testingLoss = rootNode.getLoss(validationIndices);
        }

        if(printProgress > 0) {
            printTreeSummary(trainingLoss, validationLoss, testingLoss);
            displayGqHistogram("final predicted minGq distribution",
                               Arrays.stream(rootNode.predict(trainingIndices)), true);
        }
    }
}
