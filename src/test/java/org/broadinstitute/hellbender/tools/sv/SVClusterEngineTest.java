package org.broadinstitute.hellbender.tools.sv;

import org.broadinstitute.hellbender.utils.SimpleInterval;
import org.testng.Assert;
import org.testng.annotations.BeforeTest;
import org.testng.annotations.Test;

import java.util.Arrays;
import java.util.List;

public class SVClusterEngineTest {

    private final SVClusterEngine engine = new SVClusterEngine(SVTestUtils.dict);

    @BeforeTest
    public void initializeClusterEngine() {
        engine.add(SVTestUtils.call1);
    }

    @Test
    public void testFlattenCluster() {
        //depth only and depthAndStuff have same bounds, less than call2
        final List<SVCallRecordWithEvidence> testCluster = Arrays.asList(SVTestUtils.depthOnly, SVTestUtils.depthAndStuff, SVTestUtils.call2);
        final SVCallRecordWithEvidence flattened = engine.flattenCluster(testCluster);
        Assert.assertEquals(flattened.getStart(), SVTestUtils.depthAndStuff.getStart());
        Assert.assertEquals(flattened.getEnd(), SVTestUtils.depthAndStuff.getEnd());
        //should have all the algs
        Assert.assertTrue(flattened.getAlgorithms().containsAll(SVTestUtils.depthAndStuff.getAlgorithms()));
        Assert.assertTrue(flattened.getAlgorithms().containsAll(SVTestUtils.depthOnly.getAlgorithms()));
        Assert.assertTrue(flattened.getAlgorithms().containsAll(SVTestUtils.call2.getAlgorithms()));
        //should have all the genotypes
        Assert.assertTrue(flattened.getGenotypes().containsAll(SVTestUtils.depthAndStuff.getGenotypes()));
        Assert.assertTrue(flattened.getGenotypes().containsAll(SVTestUtils.depthOnly.getGenotypes()));
        Assert.assertTrue(flattened.getGenotypes().containsAll(SVTestUtils.call2.getGenotypes()));
        //TODO: add test for insertion cluster
    }

    @Test
    public void testClusterTogether() {
        Assert.assertTrue(engine.clusterTogether(SVTestUtils.depthOnly, SVTestUtils.depthAndStuff));
        Assert.assertFalse(engine.clusterTogether(SVTestUtils.depthOnly, SVTestUtils.inversion));
        Assert.assertFalse(engine.clusterTogether(SVTestUtils.call1, SVTestUtils.call2));
        Assert.assertTrue(engine.clusterTogether(SVTestUtils.call1, SVTestUtils.overlapsCall1));
    }

    @Test
    public void testGetClusteringInterval() {
        Assert.assertTrue(engine.getClusteringInterval(SVTestUtils.leftEdgeCall, null).getStart() > 0);
        Assert.assertTrue(engine.getClusteringInterval(SVTestUtils.rightEdgeCall, null).getEnd() < SVTestUtils.chr1Length);

        final SimpleInterval littleCluster = engine.getClusteringInterval(SVTestUtils.call1, null);
        //these shouldn't really cluster, but this method doesn't check that
        final SimpleInterval totalInterval = engine.getClusteringInterval(SVTestUtils.call2, littleCluster);
        //TODO: add more quantitative checks
        //min start for combined interval should be greater than the leftmost bound, which is the start of call1
        Assert.assertTrue(totalInterval.getStart() < SVTestUtils.call1.getStart());
        //max start for combined interval should be greater than the leftmost bound, and less than the nearest event end
        Assert.assertTrue(totalInterval.getEnd() > SVTestUtils.call1.getStart());
        Assert.assertTrue(totalInterval.getEnd() > SVTestUtils.call1.getEnd());
    }

    @Test
    public void testItemsAreIdentical() {
        //same bounds, different algs
        Assert.assertTrue(engine.itemsAreIdentical(SVTestUtils.depthOnly, SVTestUtils.depthAndStuff));

        //different bounds
        Assert.assertFalse(engine.itemsAreIdentical(SVTestUtils.call1, SVTestUtils.call2));

        Assert.assertTrue(engine.itemsAreIdentical(SVTestUtils.call1, SVTestUtils.sameBoundsSampleMismatch));
    }

    @Test
    public void testDeduplicateIdenticalItems() {
        final SVCallRecordWithEvidence merged1 = engine.deduplicateIdenticalItems(Arrays.asList(SVTestUtils.depthOnly, SVTestUtils.depthAndStuff));
        Assert.assertEquals(merged1.getGenotypes().size(), 2);
        Assert.assertTrue(merged1.getGenotypes().containsAll(Arrays.asList(SVTestUtils.sample1, SVTestUtils.sample2)));
        Assert.assertEquals(merged1.getAlgorithms().size(), 2);
        Assert.assertTrue(merged1.getAlgorithms().containsAll(SVTestUtils.depthAndStuff.getAlgorithms()));
    }

    @Test
    public void testIsDepthOnlyCall() {
        Assert.assertTrue(SVClusterEngine.isDepthOnlyCall(SVTestUtils.depthOnly));
        Assert.assertFalse(SVClusterEngine.isDepthOnlyCall(SVTestUtils.depthAndStuff));
    }

    @Test
    public void testAdd() {
        //single-sample merge case, ignoring sample sets
        final SVClusterEngine temp1 = new SVClusterEngine(SVTestUtils.dict);
        temp1.add(SVTestUtils.call1);
        //force new cluster by adding a non-overlapping event
        temp1.add(SVTestUtils.call3);
        final List<SVCallRecordWithEvidence> output1 = temp1.getOutput(); //flushes all clusters
        Assert.assertEquals(output1.size(), 2);
        Assert.assertEquals(SVTestUtils.call1, output1.get(0));
        Assert.assertEquals(SVTestUtils.call3, output1.get(1));

        final SVClusterEngine temp2 = new SVClusterEngine(SVTestUtils.dict);
        temp2.add(SVTestUtils.call1);
        temp2.add(SVTestUtils.overlapsCall1);
        //force new cluster by adding a call on another contig
        temp2.add(SVTestUtils.call4_chr10);
        final List<SVCallRecordWithEvidence> output2 = temp2.getOutput();
        Assert.assertEquals(output2.size(), 2);
        //median of two items ends up being the second item here
        Assert.assertEquals(output2.get(0).getStart(), SVTestUtils.overlapsCall1.getStart());
        Assert.assertEquals(output2.get(0).getEnd(), SVTestUtils.overlapsCall1.getEnd());
        Assert.assertEquals(output2.get(1), SVTestUtils.call4_chr10);

        //checking insensitivity to sample set overlap
        final SVClusterEngine temp3 = new SVClusterEngine(SVTestUtils.dict);
        temp3.add(SVTestUtils.call1);
        temp3.add(SVTestUtils.sameBoundsSampleMismatch);
        final List<SVCallRecordWithEvidence> output3 = temp3.getOutput();
        Assert.assertEquals(output3.size(), 1);
        Assert.assertEquals(output3.get(0).getStart(), SVTestUtils.call1.getStart());
        Assert.assertEquals(output3.get(0).getEnd(), SVTestUtils.call1.getEnd());
        Assert.assertEquals(output3.get(0).getStart(), SVTestUtils.sameBoundsSampleMismatch.getStart());
        Assert.assertEquals(output3.get(0).getEnd(), SVTestUtils.sameBoundsSampleMismatch.getEnd());
    }
}