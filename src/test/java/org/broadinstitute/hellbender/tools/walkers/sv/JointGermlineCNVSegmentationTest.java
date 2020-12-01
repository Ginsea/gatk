package org.broadinstitute.hellbender.tools.walkers.sv;

import htsjdk.variant.variantcontext.Allele;
import htsjdk.variant.variantcontext.GenotypeBuilder;
import htsjdk.variant.variantcontext.VariantContext;
import htsjdk.variant.variantcontext.VariantContextBuilder;
import htsjdk.variant.vcf.VCFConstants;
import org.broadinstitute.hellbender.tools.spark.sv.utils.GATKSVVCFConstants;
import org.testng.Assert;
import org.testng.annotations.Test;

import javax.ws.rs.core.Variant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.testng.Assert.*;

public class JointGermlineCNVSegmentationTest {
    final JointGermlineCNVSegmentation walker = new JointGermlineCNVSegmentation();


    @Test
    public void testResolveVariantContexts() {


        VariantContextBuilder builder = new VariantContextBuilder();
        final GenotypeBuilder genotypeBuilder = new GenotypeBuilder("sample1");

        //make first variant sample 1
        genotypeBuilder.alleles(Arrays.asList(Allele.REF_N, GATKSVVCFConstants.DEL_ALLELE)).attribute(GATKSVVCFConstants.COPY_NUMBER_FORMAT, 1);
        builder.chr("1").start(1000).stop(2000).alleles(Arrays.asList(Allele.REF_N, GATKSVVCFConstants.DEL_ALLELE)).genotypes(genotypeBuilder.make());
        final VariantContext sample1_var1 = builder.make();

        //then second variant in sample 1, to update the sample-copy number map
        builder.start(3000).stop(5000).alleles(Arrays.asList(Allele.REF_N, GATKSVVCFConstants.DEL_ALLELE)).genotypes(genotypeBuilder.make());
        final VariantContext sample1_var2 = builder.make();

        //then an overlapping variant in sample 2
        final GenotypeBuilder gb_sample2 = new GenotypeBuilder("sample2");
        gb_sample2.alleles(Arrays.asList(Allele.REF_N, GATKSVVCFConstants.DEL_ALLELE)).attribute(GATKSVVCFConstants.COPY_NUMBER_FORMAT, 1);
        builder.start(4000).stop(5000).alleles(Arrays.asList(Allele.REF_N, GATKSVVCFConstants.DEL_ALLELE)).genotypes(genotypeBuilder.make());
        final VariantContext sample2 = builder.make();

        final List<VariantContext> resolvedVCs = walker.resolveVariantContexts(Arrays.asList(sample1_var1, sample1_var2, sample2));

        Assert.assertEquals(Integer.parseInt(resolvedVCs.get(2).getGenotype("sample1").getExtendedAttribute(GATKSVVCFConstants.COPY_NUMBER_FORMAT).toString()), 1);
    }

    @Test
    public void testUpdateGenotypes() {
    }
}