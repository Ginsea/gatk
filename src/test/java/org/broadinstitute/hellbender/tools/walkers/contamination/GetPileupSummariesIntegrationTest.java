package org.broadinstitute.hellbender.tools.walkers.contamination;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.broadinstitute.hellbender.CommandLineProgramTest;
import org.broadinstitute.hellbender.exceptions.UserException;
import org.broadinstitute.hellbender.testutils.ArgumentsBuilder;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.File;
import java.util.List;

/**
 * Created by David Benjamin on 2/16/17.
 */
public class GetPileupSummariesIntegrationTest extends CommandLineProgramTest {
    private static final File NA12878 = new File(largeFileTestDir, "CEUTrio.HiSeq.WGS.b37.NA12878.20.21.bam");

    @Test
    public void testTakuto(){
        final String home = "/Volumes/dsde_working/tsato/jukebox/contamination/calculate_contamination/improvement/";
        final File cram = new File("/Volumes/dsde_working/tsato/jukebox/contamination/bam/130401_D4-B0096_NA24695_BC10_JBX-014_C01.cram");
        final File exac = new File("/Volumes/dsde_working/tsato/jukebox/contamination/small_exac_common_3.hg38.vcf.gz");
        final File out = new File(home + "tmp.txt");
        final String hg38 = "/Volumes/seq_references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta";

        final ArgumentsBuilder args = new ArgumentsBuilder()
                .addReference(hg38)
                .addInput(cram)
                .addVCF(exac)
                .addIntervals(exac)
                .addOutput(out);
        runCommandLine(args);
    }

    @Test
    public void test() {
        final File output = createTempFile("output", ".table");

        final ArgumentsBuilder args = new ArgumentsBuilder()
                .addInput(NA12878)
                .addVCF(new File(thousandGenomes))
                .addIntervals(new File(thousandGenomes))
                .addOutput(output)
                .add(GetPileupSummaries.MAX_SITE_AF_SHORT_NAME, 0.9);

        runCommandLine(args);

        final ImmutablePair<String, List<PileupSummary>> sampleAndResult = PileupSummary.readFromFile(output);

        final List<PileupSummary> result = sampleAndResult.getRight();
        final String sample = sampleAndResult.getLeft();
        Assert.assertEquals(sample, "NA12878");

        // compare to IGV manual inspection
        final PileupSummary ps1 = result.get(0);
        Assert.assertEquals(ps1.getContig(), "20");
        Assert.assertEquals(ps1.getStart(), 10000117);
        Assert.assertEquals(ps1.getRefCount(), 35);
        Assert.assertEquals(ps1.getAltCount(), 28);
        Assert.assertEquals(ps1.getOtherAltCount(), 0);
        Assert.assertEquals(ps1.getAlleleFrequency(), 0.605);

        final PileupSummary ps2 = result.get(1);
        Assert.assertEquals(ps2.getStart(), 10000211);
        Assert.assertEquals(ps2.getRefCount(), 27);
        Assert.assertEquals(ps2.getAltCount(), 28);
        Assert.assertEquals(ps2.getAlleleFrequency(), 0.603);

        final PileupSummary ps3 = result.get(2);
        Assert.assertEquals(ps3.getStart(), 10000439);
        Assert.assertEquals(ps3.getRefCount(), 0);
        Assert.assertEquals(ps3.getAltCount(), 76);
        Assert.assertEquals(ps3.getAlleleFrequency(), 0.81);

        final PileupSummary ps4 = result.get(8);
        Assert.assertEquals(ps4.getStart(), 10001298);
        Assert.assertEquals(ps4.getRefCount(), 0);
        Assert.assertEquals(ps4.getAltCount(), 53);
        Assert.assertEquals(ps4.getOtherAltCount(), 0);
        Assert.assertEquals(ps4.getAlleleFrequency(), 0.809);

    }

    @Test(expectedExceptions = UserException.BadInput.class)
    public void testNoAFFieldInHeader() {
        final File vcfWithoutAF = new File(publicTestDir, "empty.vcf");
        final File output = createTempFile("output", ".table");

        final ArgumentsBuilder args = new ArgumentsBuilder()
                .addInput(NA12878)
                .addVCF(vcfWithoutAF)
                .addIntervals(vcfWithoutAF)
                .addOutput(output);

        runCommandLine(args);
    }

}