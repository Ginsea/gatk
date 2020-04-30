from unittest import TestCase
import io_vcf_parsing as io


class test_io_vcf_parsing(TestCase):
    def test_read_sample_segments_and_calls(self):
        intervals_vcf = '/Users/gauthier/workspaces/gatk/src/test/resources/org/broadinstitute/hellbender/tools/copynumber/gcnv-postprocess/intervals_output_SAMPLE_000.vcf.gz'
        clustered_vcf = '/Users/gauthier/workspaces/gatk/src/test/resources/org/broadinstitute/hellbender/tools/copynumber/clustering/threeSamples.vcf.gz'
        sample_name = 'SAMPLE_000'

        contig = "1"
        path = io.read_sample_segments_and_calls(intervals_vcf, clustered_vcf, sample_name, contig)
        # no segments on contig 1, so make one big interval
        self.assertTrue(len(path) == 1)

        contig = "2"
        path = io.read_sample_segments_and_calls(intervals_vcf, clustered_vcf, sample_name, contig)
        self.assertTrue(len(path) == 3)

        sample1_intervals_vcf = '/Users/gauthier/workspaces/gatk/src/test/resources/org/broadinstitute/hellbender/tools/copynumber/gcnv-postprocess/intervals_output_SAMPLE_001.vcf.gz'
        sample1_name = 'SAMPLE_001'
        contig = "2"
        path1 = io.read_sample_segments_and_calls(sample1_intervals_vcf, clustered_vcf, sample1_name, contig)
        # all samples should have the same number of intervals
        self.assertTrue(len(path) == len(path1))
