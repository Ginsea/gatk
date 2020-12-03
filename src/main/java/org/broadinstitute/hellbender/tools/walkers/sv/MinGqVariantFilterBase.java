package org.broadinstitute.hellbender.tools.walkers.sv;

import htsjdk.variant.variantcontext.*;
import htsjdk.variant.variantcontext.writer.VariantContextWriter;
import htsjdk.variant.vcf.*;
import net.minidev.json.JSONArray;
import net.minidev.json.JSONObject;
import net.minidev.json.JSONValue;
import net.minidev.json.parser.ParseException;
import org.broadinstitute.barclay.argparser.Argument;
import org.broadinstitute.hellbender.cmdline.StandardArgumentDefinitions;
import org.broadinstitute.hellbender.engine.*;
import org.broadinstitute.hellbender.exceptions.GATKException;
import org.broadinstitute.hellbender.exceptions.UserException;
import org.broadinstitute.hellbender.utils.Utils;
import org.broadinstitute.hellbender.utils.samples.PedigreeValidationType;
import org.broadinstitute.hellbender.utils.samples.SampleDBBuilder;
import org.broadinstitute.hellbender.utils.samples.Trio;

import java.io.*;
import java.math.BigDecimal;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.apache.commons.math3.util.FastMath;
import org.broadinstitute.hellbender.utils.variant.GATKVariantContextUtils;
import org.jetbrains.annotations.NotNull;

import static org.apache.commons.math3.util.FastMath.*;


/**
 * Extract matrix of properties for each variant.
 * Also extract, numVariants x numTrios x 3 tensors of allele count and genotype quality.
 * These data will be used to train a variant filter based on min GQ" (and stratified by other variant properties) that
 * maximizes the admission of variants with Mendelian inheritance pattern while omitting non-Mendelian variants.
 *
 * Derived class must implement abstract method trainFilter()
 */
public abstract class MinGqVariantFilterBase extends VariantWalker {
    @Argument(fullName=StandardArgumentDefinitions.PEDIGREE_FILE_LONG_NAME, shortName=StandardArgumentDefinitions.PEDIGREE_FILE_SHORT_NAME,
            doc="Pedigree file, necessary for \"TRAIN\" mode, ignored in \"FILTER\" mode.", optional = true)
    public File pedigreeFile = null;

    @Argument(fullName=StandardArgumentDefinitions.OUTPUT_LONG_NAME, shortName=StandardArgumentDefinitions.OUTPUT_SHORT_NAME,
              doc="Output file. In \"FILTER\" mode it is the filtered VCF. In \"TRAIN\" mode it is the saved final filtering model.")
    public File outputFile;

    @Argument(fullName="model-file", shortName="m", optional=true,
              doc="Path to saved pre-existing filter model. In \"FILTER\" mode, this is a mandatory argument."
                 +" In \"TRAIN\" mode, this is an optional argument for additional training based on an existing model.")
    public File modelFile = null;

    @Argument(fullName="truth-file", shortName="t", optional=true,
              doc="Path to JSON file with truth data. Keys are sample IDs and values are objects with key \"good\""
                 +" corresponding to a list of known true variant IDs, and key \"bad\" corresponding to a list of known"
                 +" bad variant IDs")
    public File truthFile = null;

    private enum RunMode { TRAIN, FILTER }
    @Argument(fullName="mode", doc="Mode of operation: either \"TRAIN\" or \"FILTER\"")
    public RunMode runMode;

    @Argument(fullName="keep-homvar", shortName="kh", doc="Keep homvar variants even if their GQ is less than min-GQ", optional=true)
    public boolean keepHomvar = true;

    @Argument(fullName="keep-multiallelic", shortName="km", doc="Keep multiallelic variants even if their GQ is less than min-GQ", optional=true)
    public boolean keepMultiallelic = true;

    @Argument(fullName="validation-proportion", shortName="vp", doc="Proportion of variants to set aside for cross-validation",
              optional=true, minValue=0.0, maxValue=1.0)
    public double validationProportion = 0.5;

    @Argument(fullName="report-min-gq-filter-threshold", shortName="rf", optional=true, minValue=0.0, maxValue=1.0,
              doc="Add \"" + EXCESSIVE_MIN_GQ_FILTER_KEY + "\" to FILTER if the proportion of samples with calls filtered by "
                 + MIN_GQ_KEY + " is greater than this threshold.")
    public double reportMinGqFilterThreshold = 0.005;

    @Argument(fullName="truth-weight", shortName="tw", optional=true, minValue=0.0, maxValue=1.0,
            doc="Weight for truth data in loss function.")
    public static double truthWeight = 0.5;

    @Argument(fullName="print-progress", shortName="p", doc="Print progress of fit during training", optional = true)
    public int printProgress = 1;

    static final String minSamplesToEstimateAlleleFrequencyKey = "min-samples-to-estimate-allele-frequency";
    @Argument(fullName=minSamplesToEstimateAlleleFrequencyKey, shortName="ms", optional=true,
              doc="If the VCF does not have allele frequency, estimate it from the sample population if there are at least this many samples. Otherwise throw an exception.")
    public int minSamplesToEstimateAlleleFrequency = 100;

    static final Map<String, double[]> propertyBinsMap = new HashMap<String, double[]>() {
        private static final long serialVersionUID = 0;
        {
            put(AF_PROPERTY_NAME, new double[] {0.05, 0.5});
            put(SVLEN_KEY, new double[] {500.0, 5000.0});
        }
    };

    // numVariants x numProperties matrix of variant properties
    protected Map<String, double[]> variantPropertiesMap = null;
    // numVariants x numTrios x 3 tensor of allele counts:
    protected List<int[][]> alleleCountsTensor = new ArrayList<>();
    // numVariants x numTrios x 3 tensor of genotype qualities:
    protected List<int[][]> genotypeQualitiesTensor = new ArrayList<>();
    // map from variantIndex to array of known good GQ values for that variant
    protected Map<Integer, int[]> goodVariantGqs = null;
    // map from variantIndex to array of known bad GQ values for that variant
    protected Map<Integer, int[]> badVariantGqs = null;
    // numVariants array of property-category bins
    protected int[] propertyBins = null;
    int numPropertyBins;

    protected Random randomGenerator = Utils.getRandomGenerator();

    private VariantContextWriter vcfWriter = null;

    private int numVariants;
    private int numTrios;
    private int numProperties;

    private static final String SVLEN_KEY = "SVLEN";
    private static final String EVIDENCE_KEY = "EVIDENCE";
    private static final String AF_PROPERTY_NAME = "AF";
    private static final String MIN_GQ_KEY = "MINGQ";
    private static final String EXCESSIVE_MIN_GQ_FILTER_KEY = "LOW_GQ";
    private static final String MULTIALLELIC_FILTER = "MULTIALLELIC";
    private static final String NO_EVIDENCE = "NO_EVIDENCE";
    private static final String GOOD_VARIANT_TRUTH_KEY = "good";
    private static final String BAD_VARIANT_TRUTH_KEY = "bad";

    // properties used to gather main matrix / tensors during apply()
    private Set<Trio> pedTrios = null;
    private final List<Double> alleleFrequencies = new ArrayList<>();
    private final List<String> svTypes = new ArrayList<>();
    private final List<Integer> svLens = new ArrayList<>();
    private final List<Set<String>> variantFilters = new ArrayList<>();
    private final List<Set<String>> variantEvidence = new ArrayList<>();
    private Map<String, Set<String>> goodVariantSamples = null;
    private Map<String, Set<String>> badVariantSamples = null;

    // saved initial values
    private List<String> allEvidenceTypes = null;
    private List<String> allFilterTypes = null;
    private List<String> allSvTypes = null;
    private List<String> propertyNames = null;
    private Map<String, Double> propertyBaseline = null;
    private Map<String, Double> propertyScale = null;
    private static final String ALL_EVIDENCE_TYPES_KEY = "allEvidenceTypes";
    private static final String ALL_FILTER_TYPES_KEY = "allFilterTypes";
    private static final String ALL_SV_TYPES_KEY = "allSvTypes";
    private static final String PROPERTY_NAMES_KEY = "propertyNames";
    private static final String PROPERTY_BASELINE_KEY = "propertyBaseline";
    private static final String PROPERTY_SCALE_KEY = "propertyScale";

    // train/validation split indices
    private int[] trainingIndices;
    private int[] validationIndices;

    // stats on tool actions
    private int numFilteredGenotypes;

    // properties for calculating f1 score or estimating its pseudo-derivatives
    private int[] perVariantOptimalMinGq = null;
    protected int[] maxDiscoverableMendelianAc = null;
    long numDiscoverableMendelianAc = 0;

    protected final int getNumVariants() { return numVariants; }
    protected final int getNumTrios() { return numTrios; }
    protected final int getNumProperties() { return numProperties; }
    protected final List<String> getPropertyNames() { return propertyNames; }
    protected final int[] getTrainingIndices() { return trainingIndices; }
    protected final int[] getValidationIndices() { return validationIndices; }

    /**
     * Entry-point function to initialize the samples database from input data
     */
    private void getPedTrios() {
        if(pedigreeFile == null) {
            throw new UserException.BadInput(StandardArgumentDefinitions.PEDIGREE_FILE_LONG_NAME + " must be specified in \"TRAIN\" mode");
        }
        final SampleDBBuilder sampleDBBuilder = new SampleDBBuilder(PedigreeValidationType.STRICT);
        sampleDBBuilder.addSamplesFromPedigreeFiles(Collections.singletonList(pedigreeFile));
        pedTrios = sampleDBBuilder.getFinalSampleDB().getTrios();
        if(pedTrios.isEmpty()) {
            throw new UserException.BadInput("The pedigree file must contain trios: " + pedigreeFile);
        }
    }

    private void getVariantTruthData() {
        if(truthFile == null) {
            System.out.println("No truth file specified. Not using truth data.");
            return;
        } else {
            System.out.println("Loading truth data from " + truthFile);
        }
        final JSONObject jsonObject;
        try (final FileReader fileReader = new FileReader(truthFile)){
            jsonObject = (JSONObject) JSONValue.parseStrict(fileReader);
        } catch (IOException | ParseException ioException) {
            throw new GATKException("Unable to parse JSON from inputStream", ioException);
        }
        goodVariantSamples = new HashMap<>();
        badVariantSamples = new HashMap<>();
        for(final Map.Entry<String, Object> sampleTruthEntry : jsonObject.entrySet()) {
            final String sampleId = sampleTruthEntry.getKey();
            final JSONObject sampleTruth = (JSONObject)sampleTruthEntry.getValue();
            for(final Object variantIdObj : (JSONArray)sampleTruth.get(GOOD_VARIANT_TRUTH_KEY)) {
                final String variantId = (String)variantIdObj;
                if(goodVariantSamples.containsKey(variantId)) {
                    goodVariantSamples.get(variantId).add(sampleId);
                } else {
                    goodVariantSamples.put(variantId, new HashSet<>(Collections.singleton(sampleId)) );
                }
            }
            for(final Object variantIdObj : (JSONArray)sampleTruth.get(BAD_VARIANT_TRUTH_KEY)) {
                final String variantId = (String)variantIdObj;
                if(badVariantSamples.containsKey(variantId)) {
                    badVariantSamples.get(variantId).add(sampleId);
                } else {
                    badVariantSamples.put(variantId, new HashSet<>(Collections.singleton(sampleId)));
                }
            }
        }
        // prepare to hold data for scoring
        goodVariantGqs = new HashMap<>();
        badVariantGqs = new HashMap<>();
    }

    private void initializeVcfWriter() {
        vcfWriter = createVCFWriter(outputFile);
        final Set<VCFHeaderLine> hInfo = new LinkedHashSet<>(getHeaderForVariants().getMetaDataInInputOrder());
        final String filterableVariant = (keepMultiallelic ? "biallelic " : "") + (keepHomvar ? "het " : "") + "variant";
        hInfo.add(new VCFInfoHeaderLine(MIN_GQ_KEY, 1, VCFHeaderLineType.Integer,
                             "Minimum passing GQ for each " + filterableVariant));
        hInfo.add(new VCFFilterHeaderLine(EXCESSIVE_MIN_GQ_FILTER_KEY,
                               "More than " + (100 * reportMinGqFilterThreshold) + "% of sample GTs were masked as no-call GTs due to low GQ"));
        vcfWriter.writeHeader(new VCFHeader(hInfo, getHeaderForVariants().getGenotypeSamples()));
    }

    @Override
    public void onTraversalStart() {
        loadTrainedModel();  // load model and saved properties stats
        if(runMode == RunMode.TRAIN) {
            getPedTrios();  // get trios from pedigree file
            getVariantTruthData(); // load variant truth data from JSON file
        } else {
            initializeVcfWriter();  // initialize vcfWriter and write header
            numFilteredGenotypes = 0;
        }
    }

    private static boolean mapContainsTrio(final Map<String, Integer> map, final Trio trio) {
        return map.containsKey(trio.getPaternalID()) && map.containsKey(trio.getMaternalID())
                && map.containsKey(trio.getChildID());
    }

    private static int[] getMappedTrioProperties(final Map<String, Integer> map, final Trio trio) {
        return new int[] {map.get(trio.getPaternalID()), map.get(trio.getMaternalID()), map.get(trio.getChildID())};
    }


    private boolean getIsMultiallelic(final VariantContext variantContext) {
        return variantContext.getNAlleles() > 2 || variantContext.getFilters().contains(MULTIALLELIC_FILTER);
    }

    private boolean getVariantIsFilterable(final VariantContext variantContext, final Map<String, Integer> sampleAlleleCounts) {
        final boolean maybeFilterable = !(keepMultiallelic && getIsMultiallelic(variantContext));
        if(maybeFilterable) {
            if(runMode == RunMode.FILTER) {
                // filter no matter what, because end-user may be interested in minGQ
                return true;
            } else {
                // check if any of the allele counts can be filtered so as not to train on unfilterable variants
                return pedTrios.stream()
                        .filter(trio -> mapContainsTrio(sampleAlleleCounts, trio))
                        .flatMapToInt(trio -> Arrays.stream(getMappedTrioProperties(sampleAlleleCounts, trio)))
                        .anyMatch(keepHomvar ? ac -> ac == 1 : ac -> ac > 0);
            }
        } else {
            return false;
        }
    }

    /**
     * Accumulate properties for variant matrix, and allele counts, genotype quality for trio tensors
     */
    @Override
    public void apply(VariantContext variantContext, ReadsContext readsContext, ReferenceContext ref, FeatureContext featureContext) {
        // get per-sample allele counts as a map indexed by sample ID
        final Map<String, Integer> sampleAlleleCounts = variantContext.getGenotypes().stream().collect(
                Collectors.toMap(
                        Genotype::getSampleName,
                        g -> g.getAlleles().stream().mapToInt(a -> a.isReference() ? 0 : 1).sum()
                )
        );
        if(!getVariantIsFilterable(variantContext, sampleAlleleCounts)) {
            // no need to train on unfilterable variants, and filtering is trivial
            if(runMode == RunMode.FILTER) {
                ++numVariants;
                vcfWriter.add(variantContext); // add variantContext unchanged
            }
            return;
        }
        ++numVariants;

        double alleleFrequency = variantContext.getAttributeAsDouble(VCFConstants.ALLELE_FREQUENCY_KEY, -1.0);
        if(alleleFrequency <= 0) {
            if(variantContext.getNSamples() <= minSamplesToEstimateAlleleFrequency) {
                throw new GATKException("VCF does not have " + VCFConstants.ALLELE_FREQUENCY_KEY + " annotated or enough samples to estimate it ("
                                        + minSamplesToEstimateAlleleFrequencyKey + "=" + minSamplesToEstimateAlleleFrequency + " but there are "
                                        + variantContext.getNSamples() + " samples)");
            }
            // VCF not annotated with allele frequency, guess it from allele counts
            final int numAlleles = variantContext.getGenotypes().stream().mapToInt(Genotype::getPloidy).sum();
            alleleFrequency = sampleAlleleCounts.values().stream().mapToInt(Integer::intValue).sum() / (double) numAlleles;
        }
        alleleFrequencies.add(alleleFrequency);

        final String svType = variantContext.getAttributeAsString(VCFConstants.SVTYPE, null);
        if(svType == null) {
            throw new GATKException("Missing " + VCFConstants.SVTYPE + " for variant " + variantContext.getID());
        }
        svTypes.add(svType);

        final int svLen = variantContext.getAttributeAsInt(SVLEN_KEY, Integer.MIN_VALUE);
        if(svLen == Integer.MIN_VALUE) {
            throw new GATKException("Missing " + SVLEN_KEY + " for variant " + variantContext.getID());
        }
        svLens.add(svLen);

        variantFilters.add(variantContext.getFilters());

        final Set<String> vcEvidence = Arrays.stream(
                    variantContext.getAttributeAsString(EVIDENCE_KEY, NO_EVIDENCE)
                        .replaceAll("[\\[\\] ]", "").split(",")
            ).map(ev -> ev.equals(".") ? NO_EVIDENCE : ev).collect(Collectors.toSet());
        if(vcEvidence.isEmpty()) {
            throw new GATKException("Missing " + EVIDENCE_KEY + " for variant " + variantContext.getID());
        }
        variantEvidence.add(vcEvidence);

        if(runMode == RunMode.TRAIN) {
            // get per-sample genotype qualities as a map indexed by sample ID
            final Map<String, Integer> sampleGenotypeQualities = variantContext.getGenotypes().stream().collect(
                    Collectors.toMap(Genotype::getSampleName, Genotype::getGQ)
            );

            if(goodVariantSamples != null) {
                if(goodVariantSamples.containsKey(variantContext.getID())) {
                    // Get GQ values of known good variants that are filterable
                    final int[] knownGoodGqs = goodVariantSamples.get(variantContext.getID())
                        .stream()
                        .filter(keepHomvar ?
                                sampleId -> sampleAlleleCounts.get(sampleId) == 1 :
                                sampleId -> sampleAlleleCounts.get(sampleId) > 0)
                        .mapToInt(sampleGenotypeQualities::get).toArray();
                    if(knownGoodGqs.length > 0) {
                        goodVariantGqs.put(numVariants - 1, knownGoodGqs);
                    }
                }
                if(badVariantSamples.containsKey(variantContext.getID())) {
                    // Get GQ values of known bad variants that are filterable
                    final int[] knownBadGqs = badVariantSamples.get(variantContext.getID())
                        .stream()
                        .filter(keepHomvar ?
                                sampleId -> sampleAlleleCounts.get(sampleId) == 1 :
                                sampleId -> sampleAlleleCounts.get(sampleId) > 0)
                        .mapToInt(sampleGenotypeQualities::get).toArray();
                    if(knownBadGqs.length > 0) {
                        badVariantGqs.put(numVariants - 1, knownBadGqs);
                    }
                }
            }

            // get the numTrios x 3 matrix of trio allele counts for this variant, keeping only trios where all samples
            // are present in this VariantContext
            final int[][] trioAlleleCounts = pedTrios.stream()
                    .filter(trio -> mapContainsTrio(sampleAlleleCounts, trio))
                    .map(trio -> getMappedTrioProperties(sampleAlleleCounts, trio))
                    .collect(Collectors.toList())
                    .toArray(new int[0][0]);
            alleleCountsTensor.add(trioAlleleCounts);

            // get the numTrios x 3 matrix of trio genotype qualities for this variant, keeping only trios where all samples
            // are present in this VariantContext
            final int[][] trioGenotypeQualities = pedTrios.stream()
                    .filter(trio -> mapContainsTrio(sampleGenotypeQualities, trio))
                    .map(trio -> getMappedTrioProperties(sampleGenotypeQualities, trio))
                    .collect(Collectors.toList()).toArray(new int[0][0]);
            genotypeQualitiesTensor.add(trioGenotypeQualities);
        } else {
            collectVariantPropertiesMap();
            final double[] variantProperties = propertyNames.stream().mapToDouble(name -> variantPropertiesMap.get(name)[0]).toArray();
            final int minGq = predict(variantProperties);
            vcfWriter.add(filterVariantContext(variantContext, minGq));
        }
    }

    private VariantContext filterVariantContext(final VariantContext variantContext, final int minGq) {
        final Genotype[] genotypes = new Genotype[variantContext.getNSamples()];
        int numFiltered = 0;
        int genotypeIndex = 0;
        for(final Genotype genotype : variantContext.getGenotypes()) {
            if(genotype.getGQ() >= minGq) {
                genotypes[genotypeIndex] = new GenotypeBuilder(genotype).make();
            } else {
                genotypes[genotypeIndex] = new GenotypeBuilder(genotype).alleles(GATKVariantContextUtils.noCallAlleles(genotype.getPloidy())).make();
                ++numFiltered;
            }
            ++genotypeIndex;
        }
        final VariantContextBuilder variantContextBuilder = new VariantContextBuilder(variantContext)
            .genotypes(genotypes).attribute(MIN_GQ_KEY, minGq);
        if(numFiltered > variantContext.getNSamples() * reportMinGqFilterThreshold) {
            variantContextBuilder.filter(EXCESSIVE_MIN_GQ_FILTER_KEY);
        }
        numFilteredGenotypes += numFiltered;

        return variantContextBuilder.make();
    }

    private double getBaselineOrdered(final double[] orderedValues) {
        // get baseline as median of values
        return orderedValues.length == 0 ?
                0 :
                orderedValues.length % 2 == 1 ?
                        orderedValues[orderedValues.length / 2] :
                        (orderedValues[orderedValues.length / 2 - 1] + orderedValues[orderedValues.length / 2]) / 2.0;
    }

    private double getScaleOrdered(final double[] orderedValues, final double baseline) {
        // get scale as root-mean-square difference from baseline, over central half of data (to exclude outliers)
        switch(orderedValues.length) {
            case 0:
            case 1:
                return 1.0;
            default:
                final int start = orderedValues.length / 4;
                final int stop = 3 * orderedValues.length / 4;
                double scale = 0.0;
                for(int idx = start; idx < stop; ++idx) {
                    scale += (orderedValues[idx] - baseline) * (orderedValues[idx] - baseline);
                }
                return FastMath.max(FastMath.sqrt(scale / (1 + stop - start)), 1.0e-6);
        }
    }

    private static double[] zScore(final double[] values, final double baseline, final double scale) {
        return Arrays.stream(values).map(x -> (x - baseline) / scale).toArray();
    }

    private static double[] zScore(final int[] values, final double baseline, final double scale) {
        return Arrays.stream(values).mapToDouble(x -> (x - baseline) / scale).toArray();
    }

    private static double[] zScore(final boolean[] values, final double baseline, final double scale) {
        return IntStream.range(0, values.length).mapToDouble(i -> ((values[i] ? 1 : 0) - baseline) / scale).toArray();
    }

    @SuppressWarnings("SameParameterValue")
    private double[] getPropertyAsDoubles(final String propertyName, final double[] values) {
        // Compute baseline and scale regardless, since this info is saved in model file
        if (propertyBaseline == null) {
            propertyBaseline = new HashMap<>();
        }
        if (propertyScale == null) {
            propertyScale = new HashMap<>();
        }
        if (!propertyBaseline.containsKey(propertyName)) {
            final double[] orderedValues = Arrays.stream(values).sorted().toArray();
            propertyBaseline.put(propertyName, getBaselineOrdered(orderedValues));
            propertyScale.put(propertyName,
                    getScaleOrdered(orderedValues, propertyBaseline.get(propertyName)));
        }
        if(needsZScore()) {
            return zScore(values, propertyBaseline.get(propertyName), propertyScale.get(propertyName));
        } else {
            return values;
        }
    }

    @SuppressWarnings("SameParameterValue")
    private double[] getPropertyAsDoubles(final String propertyName, final int[] values) {
        // Compute baseline and scale regardless, since this info is saved in model file
        if (propertyBaseline == null) {
            propertyBaseline = new HashMap<>();
        }
        if (propertyScale == null) {
            propertyScale = new HashMap<>();
        }
        if (!propertyBaseline.containsKey(propertyName)) {
            final double[] orderedValues = Arrays.stream(values).sorted().mapToDouble(i -> i).toArray();
            propertyBaseline.put(propertyName, getBaselineOrdered(orderedValues));
            propertyScale.put(propertyName,
                    getScaleOrdered(orderedValues, propertyBaseline.get(propertyName)));
        }
        if(needsZScore()) {
            return zScore(values, propertyBaseline.get(propertyName), propertyScale.get(propertyName));
        } else {
            return Arrays.stream(values).mapToDouble(i -> (double)i).toArray();
        }
    }

    private double[] getPropertyAsDoubles(final String propertyName, final boolean[] values) {
        // Compute baseline and scale regardless, since this info is saved in model file
        if (propertyBaseline == null) {
            propertyBaseline = new HashMap<>();
        }
        if (propertyScale == null) {
            propertyScale = new HashMap<>();
        }
        if (!propertyBaseline.containsKey(propertyName)) {
            final long numTrue = IntStream.range(0, values.length).filter(i -> values[i]).count();
            final long numFalse = values.length - numTrue;
            final double baseline = numTrue / (double) values.length;
            final double scale = numTrue == 0 || numFalse == 0 ?
                    1.0 : FastMath.sqrt(numTrue * numFalse / (values.length * (double) values.length));
            propertyBaseline.put(propertyName, baseline);
            propertyScale.put(propertyName, scale);
        }
        if(needsZScore()) {
            return zScore(values, propertyBaseline.get(propertyName), propertyScale.get(propertyName));
        } else {
            return IntStream.range(0, values.length).mapToDouble(i -> values[i] ? 1.0 : 0.0).toArray();
        }
    }

    private List<String> assignAllLabels(final List<String> labelsList, List<String> allLabels) {
        return allLabels == null ?
               labelsList.stream().sorted().distinct().collect(Collectors.toList()) :
               allLabels;
    }

    private List<String> assignAllSetLabels(final List<Set<String>> labelsList, List<String> allLabels) {
        return allLabels == null ?
               labelsList.stream().flatMap(Set::stream).sorted().distinct().collect(Collectors.toList()) :
               allLabels;
    }

    private Map<String, double[]> labelsToLabelStatus(final List<String> labels, List<String> allLabels) {
        return labelsListsToLabelStatus(
                labels.stream().map(Collections::singleton).collect(Collectors.toList()),
                allLabels
        );
    }

    private Map<String, double[]> labelsListsToLabelStatus(final List<Set<String>> labelsList, List<String> allLabels) {
        final Map<String, boolean[]> labelStatus = allLabels.stream()
                .collect(Collectors.toMap(
                        label -> label, label -> new boolean[labelsList.size()]
                ));
        int variantIdx = 0;
        for (final Set<String> variantLabels : labelsList) {
            final int idx = variantIdx; // need final or "effectively final" variable for lambda expression
                //variantLabels.forEach(label -> labelStatus.get(label)[idx] = true);
                for(final String label : variantLabels) {
                    try {
                        labelStatus.get(label)[idx] = true;
                    } catch(java.lang.NullPointerException exception) {
                        throw new GATKException(
                            "error processing " + label + ". allLabels=" + allLabels + ". labelsList.size=" + labelsList.size(),
                            exception
                        );
                    }
                }
            ++variantIdx;
        }
        return labelStatus.entrySet().stream().collect(Collectors.toMap(
                Map.Entry::getKey,
                e -> getPropertyAsDoubles(e.getKey(), e.getValue())
        ));
    }

    private void collectVariantPropertiesMap() {
        allEvidenceTypes = assignAllSetLabels(variantEvidence, allEvidenceTypes);
        allFilterTypes = assignAllSetLabels(variantFilters, allFilterTypes);
        allSvTypes = assignAllLabels(svTypes, allSvTypes);
        variantPropertiesMap = Stream.of(
            labelsListsToLabelStatus(variantEvidence, allEvidenceTypes),
            labelsListsToLabelStatus(variantFilters, allFilterTypes),
            labelsToLabelStatus(svTypes, allSvTypes),
            Collections.singletonMap(
                AF_PROPERTY_NAME, getPropertyAsDoubles(AF_PROPERTY_NAME, alleleFrequencies.stream().mapToDouble(x -> x).toArray())
            ),
            Collections.singletonMap(
                SVLEN_KEY, getPropertyAsDoubles(SVLEN_KEY, svLens.stream().mapToInt(x -> x).toArray())
            )
        ).flatMap(e -> e.entrySet().stream()).collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));

        List<String> suppliedPropertyNames = propertyNames == null ? null : new ArrayList<>(propertyNames);
        propertyNames = variantPropertiesMap.keySet().stream().sorted().collect(Collectors.toList());
        if(suppliedPropertyNames != null && !suppliedPropertyNames.equals(propertyNames)) {
            throw new GATKException("Extracted properties not compatible with existing propertyNames."
                                    + "\nSupplied: " + suppliedPropertyNames
                                    + "\nExtracted: " + propertyNames);
        }
        numProperties = propertyNames.size();

        // Clear raw columns:
        //   in FILTER mode, want to start from scratch with each variant
        //   in TRAIN mode, no reason to keep this data in memory
        variantEvidence.clear();
        variantFilters.clear();
        svTypes.clear();
        svLens.clear();
        alleleFrequencies.clear();
    }

    private void setPropertyBins() {
        propertyBins = new int[numVariants]; // guaranteed to be all zeros by Java spec
        propertyBinsMap.forEach((propertyName, rawBins) -> {
            final double[] bins = needsZScore() ?
                zScore(rawBins, propertyBaseline.get(propertyName), propertyScale.get(propertyName)) :
                rawBins;
            final int binsScale = bins.length + 1;
            final double[] properties = variantPropertiesMap.get(propertyName);
            IntStream.range(0, numVariants).forEach(variantIndex -> {
                int bin = Arrays.binarySearch(bins, properties[variantIndex]);
                if(bin < 0) { bin = ~bin; }
                propertyBins[variantIndex] = binsScale * propertyBins[variantIndex] + bin;
            });
        });
        // Now have assigned every variant an integer propertyBin, which may depend on multiple properties
        //    HOWEVER, not every code is used. Compress this so that we're not carrying around useless empty Loss objects
        final Map<Integer, Integer> binCode = new HashMap<>();
        for(int i = 0; i < numVariants; ++i) {
            final int bin = propertyBins[i];
            if(!binCode.containsKey(bin)) {
                binCode.put(bin, binCode.size());
            }
            propertyBins[i] = binCode.get(bin);
        }
        numPropertyBins = binCode.size();
    }

    private IntStream streamFilterableGq(final IntStream acStream, final IntStream gqStream) {
        final PrimitiveIterator.OfInt acIterator = acStream.iterator();
        return gqStream.filter(keepHomvar ? gc -> acIterator.nextInt() == 1 : gc -> acIterator.nextInt() > 0);
    }

    protected IntStream getCandidateMinGqs(final int rowIndex) {
        final IntStream alleleCountsStream = Arrays.stream(alleleCountsTensor.get(rowIndex)).flatMapToInt(Arrays::stream);
        final IntStream genotypeQualitiesStream = Arrays.stream(genotypeQualitiesTensor.get(rowIndex)).flatMapToInt(Arrays::stream);
        return getCandidateMinGqs(alleleCountsStream, genotypeQualitiesStream, null);
    }

    protected IntStream getCandidateMinGqs(final int[] rowIndices) {
        final IntStream alleleCountsStream = Arrays.stream(rowIndices).flatMap(
                rowIndex -> Arrays.stream(alleleCountsTensor.get(rowIndex)).flatMapToInt(Arrays::stream)
        );
        final IntStream genotypeQualitiesStream = Arrays.stream(rowIndices).flatMap(
                rowIndex -> Arrays.stream(genotypeQualitiesTensor.get(rowIndex)).flatMapToInt(Arrays::stream)
        );
        return getCandidateMinGqs(alleleCountsStream, genotypeQualitiesStream, null);
    }

    protected IntStream getCandidateMinGqs(final int[][] alleleCounts, final int[][] genotypeQualities,
                                           final Integer preexistingCandidateMinGq) {
        final IntStream alleleCountsStream = Arrays.stream(alleleCounts).flatMapToInt(Arrays::stream);
        final IntStream genotypeQualitiesStream = Arrays.stream(genotypeQualities).flatMapToInt(Arrays::stream);
        return getCandidateMinGqs(alleleCountsStream, genotypeQualitiesStream, preexistingCandidateMinGq);
    }

    protected IntStream getCandidateMinGqs(final IntStream alleleCountsStream, final IntStream genotypeQualitiesStream,
                                           final Integer preexistingCandidateMinGq) {
        // form list of candidate min GQ values that could alter filtering for this variant
        final List<Integer> candidateGq = streamFilterableGq(alleleCountsStream, genotypeQualitiesStream)
                .distinct()
                .sorted()
                .boxed()
                .collect(Collectors.toList());
        if(candidateGq.isEmpty()) {
            return null;
        }
        // consider filtering out all filterable variants by adding 1 to highest filterable GQ value
        candidateGq.add(1 + candidateGq.get(candidateGq.size() - 1));

        // find min GQ value that is redundant with preexistingCandidateMinGq
        final OptionalInt redundantIdx = preexistingCandidateMinGq == null ?
            OptionalInt.empty() :
            IntStream.range(0, candidateGq.size())
                    .filter(i -> candidateGq.get(i) >= preexistingCandidateMinGq).findFirst();

        // These candidate values are as large as they can be without altering filtering results.
        // If possible, move candidates down (midway to next change value) so that inaccuracy in predicting
        // candidateMinGq has tolerance in both directions.
        for(int index = candidateGq.size() - 2; index >= 0; --index) {
            final int gq = candidateGq.get(index);
            final int delta_gq = index == 0 ?
                    2:
                    gq - candidateGq.get(index - 1);
            if(delta_gq > 1) {
                candidateGq.set(index, gq - delta_gq / 2);
            }
        }

        // remove candidate gq that is redundant with preexistingCandidateMinGq
        if (redundantIdx.isPresent()) {
            candidateGq.remove(redundantIdx.getAsInt());
            if(candidateGq.isEmpty()) {
                return null;
            }
        }

        // return stream of primitive int
        return candidateGq.stream().mapToInt(Integer::intValue);
    }


    private boolean isMendelian(final int fatherAc, final int motherAc, final int childAc) {
        // child allele counts should not exhibit de-novo mutations nor be missing inherited homvar
        final int minAc = fatherAc / 2 + motherAc / 2;
        final int maxAc = (fatherAc > 0 ? 1 : 0) + (motherAc > 0 ? 1 : 0);
        return (minAc <= childAc) && (childAc <= maxAc);
    }

    private boolean isMendelianKeepHomvar(final int fatherAc, final int motherAc, final int childAc) {
        // child allele counts should not exhibit de-novo mutations nor be missing inherited homvar
        final int maxAc = (fatherAc > 0 ? 1 : 0) + (motherAc > 0 ? 1 : 0);
        return childAc <= maxAc;
    }

    static protected class FilterSummary {
        final int minGq;
        final long numDiscoverable;
        final long numPassed;
        final long numMendelian;
        final long numTruePositive;
        final long numFalsePositive;
        final long numFalseNegative;

        FilterSummary(final int minGq,
                      final long numDiscoverable, final long numPassed, final long numMendelian,
                      final long numTruePositive, final long numFalsePositive, final long numFalseNegative) {
            this.minGq = minGq;
            this.numDiscoverable = numDiscoverable;
            this.numPassed = numPassed;
            this.numMendelian = numMendelian;
            this.numTruePositive = numTruePositive;
            this.numFalsePositive = numFalsePositive;
            this.numFalseNegative = numFalseNegative;
        }

        static final FilterSummary EMPTY = new FilterSummary(Integer.MIN_VALUE, 0L, 0L, 0L, 0L, 0L, 0L);

        FilterSummary setMinGq(final int newMinGq) {
            return new FilterSummary(newMinGq, numDiscoverable, numPassed, numMendelian,
                                     numTruePositive, numFalsePositive, numFalseNegative);
        }

        FilterSummary shiftMinGq(final int minGqShift) {
            return setMinGq(minGq + minGqShift);
        }

        FilterSummary add(final FilterSummary other) {
            return new FilterSummary(
                minGq,
                numDiscoverable + other.numDiscoverable, numPassed + other.numPassed,
                numMendelian + other.numMendelian, numTruePositive + other.numTruePositive,
                numFalsePositive + other.numFalsePositive, numFalseNegative + other.numFalseNegative
            );
        }

        FilterSummary subtract(final FilterSummary other) {
            return new FilterSummary(
                minGq,
                numDiscoverable - other.numDiscoverable, numPassed - other.numPassed,
                numMendelian - other.numMendelian, numTruePositive - other.numTruePositive,
                numFalsePositive - other.numFalsePositive, numFalseNegative - other.numFalseNegative
            );
        }

        @Override
        public String toString() {
            return "{minGq:" + minGq + ",  numDiscoverable:" + numDiscoverable + ", numPassed:" + numPassed
                   + ", numMendelian:" + numMendelian + ", numTruePositive:" + numTruePositive
                   + ", numFalsePositive:" + numFalsePositive + ", numFalseNegative: " + numFalseNegative + "}";
        }
    }


    protected FilterSummary getFilterSummary(final int minGq, final int variantIndex) {
        final long numDiscoverable = maxDiscoverableMendelianAc[variantIndex];
        final int[][] variantAlleleCounts = alleleCountsTensor.get(variantIndex);
        final int[][] variantGenotypeQualities = genotypeQualitiesTensor.get(variantIndex);
        final int [] goodGqs = goodVariantGqs == null ? null : goodVariantGqs.getOrDefault(variantIndex, null);
        final int [] badGqs = badVariantGqs == null ? null : badVariantGqs.getOrDefault(variantIndex, null);
        return getFilterSummary(minGq, numDiscoverable, variantAlleleCounts, variantGenotypeQualities, goodGqs, badGqs);
    }

    protected BinnedFilterSummaries getBinnedFilterSummary(final int minGq, final int variantIndex) {
        return new BinnedFilterSummaries(getFilterSummary(minGq, variantIndex),
                                         propertyBins[variantIndex], numPropertyBins);
    }

    protected FilterSummary getFilterSummary(final int minGq, final long numDiscoverable,
                                             final int[][] alleleCounts, final int[][] genotypeQualities,
                                             final int[] goodGqs, final int[] badGqs) {
        long numPassed = 0;
        long numMendelian = 0;
        long numTruePositives = 0;
        long numFalsePositives = 0;
        long numFalseNegatives = 0;
        if(goodGqs != null) {
            for (final int gq : goodGqs) {
                if(gq >= minGq) {
                    ++numTruePositives;
                } else {
                    ++numFalseNegatives;
                }
            }
        }
        if(badGqs != null) {
            for(final int gq : badGqs) {
                if(gq >= minGq) {
                    ++numFalsePositives;
                }
            }
        }
        if(keepHomvar) {
            for (int trioIndex = 0; trioIndex < numTrios; ++trioIndex) {
                final int[] trioAc = alleleCounts[trioIndex];
                final int numFilterableTrio =
                        (trioAc[0] == 1 ? 1 : 0) + (trioAc[1] == 1 ? 1 : 0) + (trioAc[2] == 1 ? 1 : 0);

                if (numFilterableTrio == 0) {
                    continue;
                }
                final int[] trioGq = genotypeQualities[trioIndex];
                final int fatherAc = (trioAc[0] == 1 && trioGq[0] < minGq) ? 0 : trioAc[0];
                final int motherAc = (trioAc[1] == 1 && trioGq[1] < minGq) ? 0 : trioAc[1];
                final int childAc = (trioAc[2] == 1 && trioGq[2] < minGq) ? 0 : trioAc[2];
                // Note that we only consider an allele to have "passed" if it was in principle filterable:
                final int numPassedTrio =
                        (fatherAc == 1 ? 1 : 0) + (motherAc == 1 ? 1 : 0) + (childAc == 1 ? 1 : 0);
                if (numPassedTrio > 0) {
                    numPassed += numPassedTrio;
                    if (isMendelianKeepHomvar(fatherAc, motherAc, childAc)) {
                        numMendelian += numPassedTrio;
                    }
                }
            }
        } else {
            for (int trioIndex = 0; trioIndex < numTrios; ++trioIndex) {
                final int[] trioAc = alleleCounts[trioIndex];
                final int numFilterableTrio = trioAc[0] + trioAc[1] + trioAc[2];

                if (numFilterableTrio == 0) {
                    continue;
                }
                final int[] trioGq = genotypeQualities[trioIndex];
                final int fatherAc = trioGq[0] < minGq ? 0 : trioAc[0];
                final int motherAc = trioGq[1] < minGq ? 0 : trioAc[1];
                final int childAc = trioGq[2] < minGq ? 0 : trioAc[2];
                final int numPassedTrio = fatherAc + motherAc + childAc;
                if (numPassedTrio > 0) {
                    numPassed += numPassedTrio;
                    if (isMendelian(fatherAc, motherAc, childAc)) {
                        numMendelian += numPassedTrio;
                    }
                }
            }
        }
        return new FilterSummary(minGq, numDiscoverable, numPassed, numMendelian,
                                 numTruePositives, numFalsePositives, numFalseNegatives);
    }

    protected BinnedFilterSummaries getBinnedFilterSummary(final int minGq, final long numDiscoverable,
                                             final int[][] alleleCounts, final int[][] genotypeQualities,
                                             final int[] goodGqs, final int[] badGqs, final int propertyBin) {
        return new BinnedFilterSummaries(
            getFilterSummary(minGq, numDiscoverable, alleleCounts, genotypeQualities, goodGqs, badGqs),
            propertyBin, numPropertyBins
        );
    }

    static protected double getInheritanceF1(final long numDiscoverableMendelian, final long numMendelian, final long numPassed) {
        // calculate f1 score:
        //     f1 = 2.0 / (1.0 / recall + 1.0 / precision)
        //     recall = numMendelian / numDiscoverableMendelian
        //     precision = numMendelian / numPassed
        //     -> f1 = 2.0 * numMendelian / (numNonRef + numPassed)
        // No data to filter? -> no loss -> f1 = 1.0
        final double f1 = (numDiscoverableMendelian > 0) ?
                2.0 * (double)numMendelian / (double)(numDiscoverableMendelian + numPassed) :
                1.0;
        if(f1 < 0 || f1 > 1) {
            throw new GATKException("f1 out of range(f1=" + f1 + "). numDiscoverableMendelian=" + numDiscoverableMendelian
                                    + ", numMendelian=" + numMendelian + ", numPassed=" + numPassed);
        }
        return f1;
    }

    static protected double getTruthF1(final long numTruePositive, final long numFalsePositive, final long numFalseNegative) {
        // NOTE: calls are compared against list of true variants and false variants
        // calculate f1 score:
        //     f1 = 2.0 / (1.0 / recall + 1.0 / precision)
        //     recall = numTruePositive / (numTruePositive + numFalseNegative)
        //     precision = numTruePositive / (numTruePositive + numFalsePositive)
        //     -> f1 = 2.0 * numTruePositive / (2 * numTruePositive + numFalseNegative + numFalsePositive)
        // No data to filter? -> no loss -> f1 = 1.0
        final double f1 = numFalseNegative + numFalsePositive > 0 ?
            2.0 * numTruePositive / (2.0 * numTruePositive + numFalseNegative + numFalsePositive) :
            1.0;
        if(f1 < 0 || f1 > 1) {
            throw new GATKException("f1 out of range(f1=" + f1 + "). numTruePositive=" + numTruePositive
                    + ", numFalsePositive=" + numFalsePositive + ", numFalseNegative=" + numFalseNegative);
        }
        return f1;
    }

    static protected class Loss implements Comparable<Loss> {
        final double inheritanceLoss;
        final double truthLoss;
        final static double truthWeight = MinGqVariantFilterBase.truthWeight;
        final static double inheritanceWeight = 1.0 - truthWeight;

        Loss(final double inheritanceLoss, final double truthLoss) {
            this.inheritanceLoss = inheritanceLoss;
            this.truthLoss = truthLoss;
        }

        Loss(final FilterSummary filterSummary) {
            this(
            1.0 - getInheritanceF1(filterSummary.numDiscoverable, filterSummary.numMendelian, filterSummary.numPassed),
            1.0 - getTruthF1(filterSummary.numTruePositive, filterSummary.numFalsePositive, filterSummary.numFalseNegative)
            );
        }

        Loss(final Loss copyLoss) {
            this(copyLoss.inheritanceLoss, copyLoss.truthLoss);
        }

        Loss(final Collection<FilterSummary> filterSummaries) {
            this(
                filterSummaries.stream()
                    .map(Loss::new)
                    .reduce(Loss::add).orElse(EMPTY)
                    .divide( max(filterSummaries.size(), 1) )
            );
        }

        static Loss add(final Loss lossA, final Loss lossB) {
            return new Loss(lossA.inheritanceLoss + lossB.inheritanceLoss, lossA.truthLoss + lossB.truthLoss);
        }

        Loss divide(final int scale) {
            return new Loss(inheritanceLoss / scale, truthLoss / scale);
        }

//        public boolean ge(final Loss other) {
//            return (inheritanceLoss >= other.inheritanceLoss) && (truthLoss >= other.truthLoss);
//        }
//
//        public boolean gt(final Loss other) {
//            return this.ge(other) && (inheritanceLoss > other.inheritanceLoss) || (truthLoss > other.truthLoss);
//        }
//
//        public boolean le(final Loss other) {
//            return (inheritanceLoss <= other.inheritanceLoss) && (truthLoss <= other.truthLoss);
//        }
//
//        public boolean lt(final Loss other) {
//            return this.le(other) && (inheritanceLoss < other.inheritanceLoss) || (truthLoss < other.truthLoss);
//        }

        public boolean ge(final Loss other) {
            return toDouble() >= other.toDouble();
        }

        public boolean gt(final Loss other) {
            return toDouble() > other.toDouble();
        }

        public boolean le(final Loss other) {
            return toDouble() <= other.toDouble();
        }

        public boolean lt(final Loss other) {
            return toDouble() < other.toDouble();
        }

        @Override
        public int compareTo(final @NotNull Loss other) {
            if(this.lt(other)) {
                return -1;
            } else if(this.gt(other)) {
                return 1;
            } else {
                return 0;
            }
        }

        @Override
        public String toString() { return "{inherit:" + inheritanceLoss + ",truth:" + truthLoss + "}"; }

        double toDouble() { return inheritanceWeight * inheritanceLoss + truthWeight * truthLoss; }
        float toFloat() { return (float)toDouble(); }

        static final Loss POSITIVE_INFINITY = new Loss(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY);
        static final Loss NaN = new Loss(Double.NaN, Double.NaN);
        static final Loss EMPTY = new Loss(FilterSummary.EMPTY);
    }

    static protected class BinnedFilterSummaries extends ArrayList<FilterSummary> {
        private static final long serialVersionUID = 0;

        BinnedFilterSummaries(final Collection<FilterSummary> filterSummaries) {
            super(filterSummaries);
        }

        BinnedFilterSummaries(final FilterSummary filterSummary, final int binIndex, final int numBins) {
            this(
                IntStream.range(0, numBins)
                    .mapToObj(i -> i == binIndex ? filterSummary : FilterSummary.EMPTY)
                    .collect(Collectors.toList())
            );
        }

        BinnedFilterSummaries add(final BinnedFilterSummaries other) {
            return this.equals(BinnedFilterSummaries.EMPTY) ?
                other :
                other.equals(BinnedFilterSummaries.EMPTY) ?
                    this :
                    new BinnedFilterSummaries(
                        IntStream.range(0, this.size()).mapToObj(
                            i -> this.get(i).add(other.get(i))).collect(Collectors.toList()
                        )
                    );
        }

        BinnedFilterSummaries subtract(final BinnedFilterSummaries other) {
            return other.equals(BinnedFilterSummaries.EMPTY) ?
                this :
                this.equals(BinnedFilterSummaries.EMPTY) ?
                    new BinnedFilterSummaries(
                            other.stream()
                                .map(FilterSummary.EMPTY::subtract)
                                .collect(Collectors.toList())
                    ) :
                    new BinnedFilterSummaries(
                            IntStream.range(0, this.size()).mapToObj(
                                    i -> this.get(i).subtract(other.get(i))).collect(Collectors.toList()
                            )
                    );
        }

        int getMinGq() {
            // Different bins may be empty (set to Integer.MIN_VALUE)
            // Thus minGq is either inconsistent (summed over multiple variants) or the maximum over all bins.
            return this.stream().mapToInt(filterSummary -> filterSummary.minGq).max().orElseThrow(RuntimeException::new);
        }

        BinnedFilterSummaries setMinGq(final int newMinGq) {
            return new BinnedFilterSummaries(
                this.stream().map(filterSummary -> filterSummary.setMinGq(newMinGq)).collect(Collectors.toList())
            );
        }

        static BinnedFilterSummaries EMPTY = new BinnedFilterSummaries(FilterSummary.EMPTY, 0, 1);
    }

    static protected class FilterQuality extends BinnedFilterSummaries implements Comparable<FilterQuality> {
        private static final long serialVersionUID = 0;
        final Loss loss;

        FilterQuality(final Collection<FilterSummary> filterSummaries) {
            super(filterSummaries);
            this.loss = new Loss(filterSummaries);
        }

        FilterQuality(final FilterSummary filterSummary) {
            this(Collections.singletonList(filterSummary));
        }


        @Override
        public int compareTo(final @NotNull FilterQuality other) { return this.loss.compareTo(other.loss); }

        static final FilterQuality EMPTY = new FilterQuality(FilterSummary.EMPTY);
    }

    protected FilterQuality getOptimalVariantMinGq(final int variantIndex, final BinnedFilterSummaries backgroundFilterSummary) {
        final IntStream candidateMinGqs = getCandidateMinGqs(variantIndex);
        if(candidateMinGqs == null) {
            // minGq doesn't matter for this row, so return previous optimal filter or trivial filter
            return backgroundFilterSummary == null ? FilterQuality.EMPTY : new FilterQuality(backgroundFilterSummary);
        }

        final long numDiscoverable = maxDiscoverableMendelianAc[variantIndex];
        final int[][] variantAlleleCounts = alleleCountsTensor.get(variantIndex);
        final int[][] variantGenotypeQualities = genotypeQualitiesTensor.get(variantIndex);
        final int [] goodGqs = goodVariantGqs == null ? null : goodVariantGqs.getOrDefault(variantIndex, null);
        final int [] badGqs = badVariantGqs == null ? null : badVariantGqs.getOrDefault(variantIndex, null);
        final int propertyBin = propertyBins[variantIndex];
        if(backgroundFilterSummary == null) {
            // doing optimization only considering loss of each individual variant
            return candidateMinGqs.parallel()
                    .mapToObj(minGq -> new FilterQuality(
                            getBinnedFilterSummary(minGq, numDiscoverable, variantAlleleCounts, variantGenotypeQualities,
                                                   goodGqs, badGqs, propertyBin)
                        ))
                    .min(FilterQuality::compareTo)
                    .orElseThrow(RuntimeException::new);
        } else {
            // doing optimization considering overall loss
            return candidateMinGqs
                    .parallel()
                    .mapToObj(minGq -> new FilterQuality(
                            getBinnedFilterSummary(minGq, numDiscoverable, variantAlleleCounts, variantGenotypeQualities,
                                                   goodGqs, badGqs, propertyBin)
                            .add(backgroundFilterSummary)
                    ))
                    .min(FilterQuality::compareTo)
                    .orElseThrow(RuntimeException::new);
        }
    }

    protected BinnedFilterSummaries getBackgroundFilterSummary(
            final int[] optimizingIndices, final int[] trainingIndices,
            int[] minGqs
    ) {
        // Make empty summary
        BinnedFilterSummaries filterSummary = BinnedFilterSummaries.EMPTY;
        if(trainingIndices == null || trainingIndices.length == 0 || trainingIndices == optimizingIndices) {
            // Not using any background, return trivial background summary
            return filterSummary;
        }
        if(minGqs == null) {
            throw new GATKException("If using non-trivial trainingIndices, must pass non-null minGqs with length equal to trainingIndices.length");
        }

        int optimizeIndex = 0;
        int nextRow = optimizingIndices == null ? Integer.MAX_VALUE : optimizingIndices[optimizeIndex];
        if(nextRow < trainingIndices[0]) {
            throw new GATKException("optimizingIndices start before training set");
        }
        // Add results at training indices not in the eval set
        for (int i = 0; i < trainingIndices.length; ++i) {
            final int trainingIndex = trainingIndices[i];
            if (nextRow > trainingIndex) {
                // This index is not in the eval set, add it to background
                filterSummary = filterSummary.add(getBinnedFilterSummary(minGqs[i], trainingIndex));
            } else {
                // This index is in the optimize set, skip it and point to next optimize index
                ++optimizeIndex;
                nextRow = optimizeIndex < optimizingIndices.length ?
                    optimizingIndices[optimizeIndex] :
                    Integer.MAX_VALUE;
            }
        }

        return filterSummary;
    }

    /**
     * Optimize minGq as a constant value on optimizeIndices,
     * @param optimizeIndices
     * @param trainingIndices
     * @param minGqs
     * @return
     */
    protected FilterQuality getOptimalMinGq(final int[] optimizeIndices,
                                            final int[] trainingIndices, final int[] minGqs) {
        if(optimizeIndices.length == 0) {
            throw new GATKException("Can't get optimalMinGq from empty rowIndices");
        }
        IntStream candidateMinGqs = getCandidateMinGqs(optimizeIndices);
        if(candidateMinGqs == null) {
            // minGq doesn't matter for these rows, just return something
            candidateMinGqs = IntStream.of(Integer.MIN_VALUE);
        }

        final BinnedFilterSummaries backgroundFilterSummary = getBackgroundFilterSummary(
                optimizeIndices, trainingIndices, minGqs
        );

        // For each candidate minGq:
        //      add backgroundFilterSummary to FilterSummaries from optimizeIndices
        //      convert to FilterQuality (calculate Loss)
        // Return FilterQuality corresponding to best Loss
        return candidateMinGqs
                .parallel()
                .mapToObj(
                    candidateMinGq -> new FilterQuality(
                        Arrays.stream(optimizeIndices)
                            .mapToObj(evalIndex -> getBinnedFilterSummary(candidateMinGq, evalIndex))
                            .reduce(backgroundFilterSummary.setMinGq(candidateMinGq), BinnedFilterSummaries::add)
                    )
                )
                .min(FilterQuality::compareTo)
                .orElseThrow(() -> new GATKException("Could not find optimal minGq filter. This is a bug."));
    }

    private void setMaxDiscoverableMendelianAc() {
        maxDiscoverableMendelianAc = new int [numVariants];
        numDiscoverableMendelianAc = 0;
        for(int variantIndex = 0; variantIndex < numVariants; ++variantIndex) {
            final int[][] variantAlleleCounts = alleleCountsTensor.get(variantIndex);
            final int[][] variantGenotypeQualities = genotypeQualitiesTensor.get(variantIndex);
            final IntStream candidateMinGq = getCandidateMinGqs(variantAlleleCounts, variantGenotypeQualities, null);

            maxDiscoverableMendelianAc[variantIndex] = candidateMinGq == null ? 0 :
                candidateMinGq.parallel().map(
                        minGq -> (int)getFilterSummary(minGq, 0, variantAlleleCounts, variantGenotypeQualities, null, null).numMendelian
                ).max().orElse(0);
            numDiscoverableMendelianAc += maxDiscoverableMendelianAc[variantIndex];
        }
    }

    private void setPerVariantOptimalMinGq() {
        // Get intial optimal filter qualities, optimizing each variant separately
        // Collect total summary stats, store min GQ
        perVariantOptimalMinGq = new int [numVariants];
        BinnedFilterSummaries overallSummary = BinnedFilterSummaries.EMPTY;
        final BinnedFilterSummaries[] filterSummaries = new BinnedFilterSummaries[numVariants];
        for(int variantIndex = 0; variantIndex < numVariants; ++variantIndex) {
            final FilterQuality greedyFilter = getOptimalVariantMinGq(variantIndex, null);
            filterSummaries[variantIndex] = greedyFilter;
            overallSummary = overallSummary.add(greedyFilter);
            perVariantOptimalMinGq[variantIndex] = greedyFilter.getMinGq();
        }

        // Iteratively improve filters, optimizing for OVERALL loss
        boolean anyImproved = true;
        int numIterations = 0;
        Loss previousLoss = new Loss(overallSummary);
        while(anyImproved) {
            ++numIterations;
            anyImproved = false;
            for(int variantIndex = 0; variantIndex < numVariants; ++variantIndex) {
                final BinnedFilterSummaries previousFilter = filterSummaries[variantIndex];
                final BinnedFilterSummaries backgroundFilter = overallSummary.subtract(previousFilter);
                final FilterQuality greedyFilter = getOptimalVariantMinGq(variantIndex, backgroundFilter);
                perVariantOptimalMinGq[variantIndex] = greedyFilter.getMinGq();
                filterSummaries[variantIndex] = greedyFilter.subtract(backgroundFilter);
                overallSummary = greedyFilter;

                if(greedyFilter.loss.lt(previousLoss)) {
                    anyImproved = true;
                } else if(greedyFilter.loss.gt(previousLoss)) {
                    throw new GATKException("Loss increased. This is a bug. previousLoss=" + previousLoss
                                            + ", greedyFilter.loss=" + greedyFilter.loss);
                }
                previousLoss = greedyFilter.loss;
            }
            System.out.println("Iteration " + numIterations + ": loss=" + previousLoss);
        }
        displayHistogram("Optimal minGq histogram", Arrays.stream(perVariantOptimalMinGq), true);
    }

    final int[] take(final int[] values, final int[] indices) {
        return Arrays.stream(indices).map(i -> values[i]).toArray();
    }

    final double[] take(final double[] values, final int[] indices) {
        return Arrays.stream(indices).mapToDouble(i -> values[i]).toArray();
    }

    protected Map<String, double[]> getVariantProperties(final int [] rowIndices) {
        final Map<String, double[]> subMap = new HashMap<>();
        for(final Map.Entry<String, double[]> entry : variantPropertiesMap.entrySet()) {
            subMap.put(entry.getKey(), take(entry.getValue(), rowIndices));
        }
        return subMap;
    }

    protected int[] getPerVariantOptimalMinGq(final int[] variantIndices) {
        return take(perVariantOptimalMinGq, variantIndices);
    }

    @SuppressWarnings("SameParameterValue")
    protected void displayHistogram(final String description, final IntStream intStream, boolean binValues) {
        final Map<Integer, Integer> rawValuesMap = new HashMap<>();
        intStream.forEach(gq -> {
            if (rawValuesMap.containsKey(gq)) {
                rawValuesMap.put(gq, 1 + rawValuesMap.get(gq));
            } else {
                rawValuesMap.put(gq, 1);
            }
        });
        if(rawValuesMap.size() == 0) {
            System.out.println(description + ": no data");
        }
        final int minGqValue = rawValuesMap.keySet().stream().min(Integer::compareTo).orElseThrow(RuntimeException::new);
        final int maxGqValue = rawValuesMap.keySet().stream().max(Integer::compareTo).orElseThrow(RuntimeException::new);

        final Map<Integer, Integer> displayValuesMap;
        if(binValues) {
            displayValuesMap = new HashMap<>();
            rawValuesMap.forEach((gq, numGq) -> {
                final int binGq;
                if (gq == 0) {
                    binGq = gq;
                } else {
                    final int magnitude = (int) Math.pow(10.0, Math.floor(Math.log10(Math.abs(gq))));
                    binGq = magnitude * (gq / magnitude);
                }

                if (displayValuesMap.containsKey(binGq)) {
                    displayValuesMap.put(binGq, numGq + displayValuesMap.get(binGq));
                } else {
                    displayValuesMap.put(binGq, numGq);
                }
            });
        } else {
            displayValuesMap = rawValuesMap;
        }
        System.out.println(description + ":");
        System.out.println("min=" + minGqValue + ", max=" + maxGqValue);
        displayValuesMap.keySet()
            .stream()
            .sorted()
            .forEach(minGq -> System.out.println(minGq + ": " + displayValuesMap.get(minGq)));
    }

    void printDebugInfo() {
        System.out.println("########################################");
        System.out.println("numVariants: " + numVariants);
        System.out.println("numTrios: " + numTrios);
        System.out.println("numProperties: " + numProperties);
        System.out.println("index\tpropertyName\tpropertyBaseline\tpropertyScale");
        int idx = 0;
        for(final String propertyName : propertyNames) {
            System.out.println(idx + "\t" + propertyName + "\t" + propertyBaseline.get(propertyName) + "\t" + propertyScale.get(propertyName));
            ++idx;
        }
        System.out.println("filter types:");
        idx = 0;
        for(final String filterType : allFilterTypes) {
            System.out.println(idx + "\t" + filterType);
            ++idx;
        }
        System.out.println("evidence types:");
        idx = 0;
        for(final String evidenceType : allEvidenceTypes) {
            System.out.println(idx + "\t" + evidenceType);
            ++idx;
        }
        System.out.println("sv types:");
        idx = 0;
        for(final String svType : allSvTypes) {
            System.out.println(idx + "\t" + svType);
            ++idx;
        }

        final IntStream acStream = alleleCountsTensor.stream().flatMapToInt(
                acArr -> Arrays.stream(acArr).flatMapToInt(Arrays::stream)
        );
        final IntStream gqStream = genotypeQualitiesTensor.stream().flatMapToInt(
                gqArr -> Arrays.stream(gqArr).flatMapToInt(Arrays::stream)
        );
        displayHistogram("Filterable alleles Gq histogram:",
                            streamFilterableGq(acStream, gqStream),true);

        System.out.println("########################################");
    }

    protected Loss getLoss(final int[] minGq, final int[] variantIndices) {
        if(minGq.length != variantIndices.length) {
            throw new GATKException(
                    "Length of minGq (" + minGq.length + ") does not match length of variantIndices (" + variantIndices.length + ")"
            );
        }
        return new Loss(
            IntStream.range(0, minGq.length)
                .mapToObj(i -> getBinnedFilterSummary(minGq[i], variantIndices[i]))
                .reduce(BinnedFilterSummaries::add).orElse(FilterQuality.EMPTY)
        );
    }

    private void setTrainingAndValidationIndices() {
        final int numValidationIndices = (int)round(validationProportion * numVariants);
        final List<Integer> shuffleIndices = IntStream.range(0, numVariants).boxed().collect(Collectors.toList());
        Collections.shuffle(shuffleIndices, randomGenerator);

        validationIndices = shuffleIndices.subList(0, numValidationIndices).stream()
            .sorted().mapToInt(Integer::intValue).toArray();
        trainingIndices = shuffleIndices.subList(numValidationIndices, numVariants).stream()
            .sorted().mapToInt(Integer::intValue).toArray();
    }

    private void saveTrainedModel() {
        try (final OutputStream outputStream = new FileOutputStream(modelFile)) {
            final OutputStream unclosableOutputStream = new FilterOutputStream(outputStream) {
                @Override
                public void close() {
                    // don't close the stream in one of the subroutines
                }
            };
            saveDataPropertiesSummaryStats(unclosableOutputStream);
            unclosableOutputStream.write("\n".getBytes());
            saveModel(unclosableOutputStream);
        } catch(IOException ioException) {
            throw new GATKException("Error saving modelFile " + modelFile, ioException);
        }
    }

    private void saveDataPropertiesSummaryStats(final OutputStream outputStream) {
        final JSONArray evidenceTypes = new JSONArray(); evidenceTypes.addAll(allEvidenceTypes);
        final JSONArray filterTypes = new JSONArray(); filterTypes.addAll(allFilterTypes);
        final JSONArray svTypes = new JSONArray(); svTypes.addAll(allSvTypes);
        final JSONArray propNames = new JSONArray();
        final JSONArray propBase = new JSONArray();
        final JSONArray propScale = new JSONArray();
        for(final String propName : propertyNames) {
            propNames.add(propName);
            propBase.add(propertyBaseline.get(propName));
            propScale.add(propertyScale.get(propName));
        }
        final JSONObject jsonObject = new JSONObject();
        jsonObject.put(ALL_EVIDENCE_TYPES_KEY, evidenceTypes);
        jsonObject.put(ALL_FILTER_TYPES_KEY, filterTypes);
        jsonObject.put(ALL_SV_TYPES_KEY, svTypes);
        jsonObject.put(PROPERTY_NAMES_KEY, propNames);
        jsonObject.put(PROPERTY_BASELINE_KEY, propBase);
        jsonObject.put(PROPERTY_SCALE_KEY, propScale);

        try {
            outputStream.write(jsonObject.toJSONString().getBytes());
        } catch(IOException ioException) {
            throw new GATKException("Error saving data summary json", ioException);
        }
    }

    private void loadTrainedModel() {
        if(!modelFile.exists()) {
            if(runMode == RunMode.FILTER) {
                throw new GATKException("mode=FILTER, but trained model file does not exist.");
            }
            return;
        }
        try (final InputStream inputStream = new FileInputStream(modelFile)) {
            final InputStream unclosableInputStream = new FilterInputStream(inputStream) {
                @Override
                public void close() {
                    // don't close the stream in one of the subroutines
                }
            };
            loadDataPropertiesSummaryStats(unclosableInputStream);
            loadModel(unclosableInputStream );
        } catch (IOException ioException) {
            throw new GATKException("Error loading modelFile " + modelFile, ioException);
        }
    }

    protected double getDoubleFromJSON(final Object jsonObject) {
        if(jsonObject instanceof Double) {
            return (Double) jsonObject;
        } else if(jsonObject instanceof BigDecimal) {
            return ((BigDecimal)jsonObject).doubleValue();
        } else {
            throw new GATKException("Unknown conversion to double for " + jsonObject.getClass().getName());
        }
    }

    protected List<String> getStringListFromJSON(final Object jsonObject) {
        return ((JSONArray)jsonObject).stream().map(o -> (String)o).collect(Collectors.toList());
    }

    //private static final String ALL_EVIDENCE_TYPES_KEY = "allEvidenceTypes";
    //    private static final String ALL_FILTER_TYPES_KEY = "allFilterTypes";
    //    private static final String ALL_SV_TYPES_KEY = "allSvTypes";
    private void loadDataPropertiesSummaryStats(final InputStream inputStream) {
        final JSONObject jsonObject;
        try {
            jsonObject = (JSONObject) JSONValue.parseStrict(inputStream);
        } catch (IOException | ParseException ioException) {
            throw new GATKException("Unable to parse JSON from inputStream", ioException);
        }
        allEvidenceTypes = getStringListFromJSON(jsonObject.get(ALL_EVIDENCE_TYPES_KEY));
        allFilterTypes = getStringListFromJSON(jsonObject.get(ALL_FILTER_TYPES_KEY));
        allSvTypes = getStringListFromJSON(jsonObject.get(ALL_SV_TYPES_KEY));
        final JSONArray propNames = ((JSONArray) jsonObject.get(PROPERTY_NAMES_KEY));
        final JSONArray propBase = ((JSONArray) jsonObject.get(PROPERTY_BASELINE_KEY));
        final JSONArray propScale = ((JSONArray) jsonObject.get(PROPERTY_SCALE_KEY));
        propertyNames = new ArrayList<>();
        propertyBaseline = new HashMap<>();
        propertyScale = new HashMap<>();
        for (int idx = 0; idx < propNames.size(); ++idx) {
            final String propName = (String) propNames.get(idx);
            propertyNames.add(propName);
            propertyBaseline.put(propName, getDoubleFromJSON(propBase.get(idx)));
            propertyScale.put(propName, getDoubleFromJSON(propScale.get(idx)));
        }
    }

    private byte[] modelCheckpoint = null;

    protected void saveModelCheckpoint() {
        final ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        saveModel(outputStream);
        modelCheckpoint = outputStream.toByteArray();
    }

    protected void loadModelCheckpoint() {
        final ByteArrayInputStream inputStream = new ByteArrayInputStream(modelCheckpoint);
        loadModel(inputStream);
    }

    protected abstract boolean needsZScore();
    protected abstract int predict(final double[] variantProperties);
    protected abstract void trainFilter();
    protected abstract void saveModel(final OutputStream outputStream);
    protected abstract void loadModel(final InputStream inputStream);

    @Override
    public Object onTraversalSuccess() {
        if(runMode == RunMode.TRAIN) {
            if(numVariants == 0) {
                throw new GATKException("No variants contained in vcf: " + drivingVariantFile);
            }
            numTrios = alleleCountsTensor.get(0).length; // note: this is number of complete trios in intersection of pedigree file and VCF
            if(numTrios == 0) {
                throw new UserException.BadInput("There are no trios from the pedigree file that are fully represented in the vcf");
            }
            if(goodVariantSamples != null) {  // these aren't needed anymore, free memory
                goodVariantSamples.clear();
                goodVariantSamples = null;
                badVariantSamples.clear();
                badVariantSamples = null;
            }

            collectVariantPropertiesMap();
            setPropertyBins();
            setTrainingAndValidationIndices();
            setMaxDiscoverableMendelianAc();
            setPerVariantOptimalMinGq();

            printDebugInfo();

            trainFilter();
            saveTrainedModel();
        } else {
            System.out.println("Filtered " + numFilteredGenotypes + " genotypes in " + numVariants + " variants.");
        }
        return null;
    }
}
