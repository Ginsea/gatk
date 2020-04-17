package org.broadinstitute.hellbender.tools.walkers.sv;

import htsjdk.variant.variantcontext.VariantContext;
import htsjdk.variant.vcf.VCFHeader;
import org.apache.commons.lang3.tuple.Pair;
import org.broadinstitute.hellbender.CommandLineProgramTest;
import org.broadinstitute.hellbender.GATKBaseTest;
import org.broadinstitute.hellbender.testutils.ArgumentsBuilder;
import org.broadinstitute.hellbender.testutils.VariantContextTestUtils;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.io.File;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.broadinstitute.hellbender.GATKBaseTest.toolsTestDir;
import static org.broadinstitute.hellbender.testutils.BaseTest.createTempFile;
import static org.testng.Assert.*;

public class JointCNVSegmentationIntegrationTest extends CommandLineProgramTest {
    private static final File TEST_SUB_DIR = new File(toolsTestDir, "copynumber/gcnv-postprocess");

    private static final List<File> SEGMENTS_VCF_CORRECT_OUTPUTS = Arrays.asList(
    new File(TEST_SUB_DIR, "segments_output_SAMPLE_000.vcf"),
    new File(TEST_SUB_DIR, "segments_output_SAMPLE_001.vcf"),
    new File(TEST_SUB_DIR, "segments_output_SAMPLE_002.vcf"));

    @DataProvider
    public Object[][] postprocessOutputs() {
        return new Object[][] {
               new Object[]{SEGMENTS_VCF_CORRECT_OUTPUTS}
        };
    }

    @Test(dataProvider = "postprocessOutputs")
    public void testThreeGCNVSamples(List<File> inputVcfs) {
        final File output = createTempFile("threeSamples", ".vcf");

        final ArgumentsBuilder args = new ArgumentsBuilder()
                .addOutput(output)
                .addReference(GATKBaseTest.b37Reference);

        inputVcfs.forEach(vcf -> args.addVCF(vcf));

        runCommandLine(args, JointCNVSegmentation.class.getSimpleName());

        final Pair<VCFHeader, List<VariantContext>> withQStreshold = VariantContextTestUtils.readEntireVCFIntoMemory(output.getAbsolutePath());
        Assert.assertEquals(withQStreshold.getRight().size(), 7);

        final File output2 = createTempFile("threeSamples.noQSthreshold",".vcf");

        final ArgumentsBuilder args2 = new ArgumentsBuilder()
                .addOutput(output2)
                .addReference(GATKBaseTest.b37Reference)
                .add(JointCNVSegmentation.MIN_QUALITY_LONG_NAME, 0);
        inputVcfs.forEach(vcf -> args2.addVCF(vcf));

        runCommandLine(args2, JointCNVSegmentation.class.getSimpleName());

        final Pair<VCFHeader, List<VariantContext>> withoutQStreshold = VariantContextTestUtils.readEntireVCFIntoMemory(output2.getAbsolutePath());
        Assert.assertEquals(withoutQStreshold.getRight().size(), 8);
        //extra variant at X:227988

        //another test to make sure adjacent events with different copy numbers don't get merged/defragmented?
    }
}