import logging
import vcf
from typing import List, Tuple

_logger = logging.getLogger(__name__)


# TODO: for now I'm going to do the lazy thing and just traverse the VCF each time for each sample
def read_sample_segments_and_calls(intervals_vcf: str,
                                   clustered_vcf: str,
                                   sample_name: str,
                                   contig: str) -> List[Tuple[int, int, int]]:
    """
    Get the segmentation "path" to use for calculating qualities based on the VCF with clustered breakpoints
    :param intervals_vcf:
    :param clustered_vcf:
    :param sample_name:
    :param contig:
    :return: {copy number, start index, stop index (inclusive)}
    """
    intervals = vcf.Reader(filename=intervals_vcf)
    segments = vcf.Reader(filename=clustered_vcf)

    tracker_position = -1
    path = []
    segment_start_index = 0
    segment_end_index = 0
    segment_copy_number = 2

    try:
        intervals_iter = iter(intervals.fetch(contig))
    except ValueError:
        intervals_rec = None
    else:
        intervals_rec = next(intervals_iter)

    try:
        segments_iter = iter(segments.fetch(contig))
    except ValueError:
        segments_rec = None
    else:
        segments_rec = next(segments_iter)

    while intervals_rec is not None:
        # A record corresponds to [CHROM,POS,REF,ALT], if the same it checks the metrics differences.
        # start a (variant) segment, end previous segment
        if segments_rec is not None and segments_rec.POS <= intervals_rec.POS:
            intervals_copy_number = try_getting_format_attribute(intervals_rec, sample_name, 'CN')
            segment_copy_number = try_getting_format_attribute(segments_rec, sample_name, 'CN')
            # If the copy number is the same between the records
            if intervals_copy_number != segment_copy_number:
                print('maybe these should not have been merged if they have different copy numbers')
        # continue a segment
        elif segments_rec is not None and segments_rec.POS > try_getting_info_attribute(intervals_rec, 'END'):
            segment_end_index += 1
        # segments are used up
        else:
            segment_end_index += 1
        # always advance the intervals
        try:
            intervals_rec = next(intervals_iter)
        except StopIteration:
            segment_end_index -= 1
            path.append((segment_copy_number, segment_start_index, segment_end_index))
            break
        # advance the segments if necessary
        if segments_rec is not None and try_getting_info_attribute(segments_rec, 'END') < intervals_rec.POS:
            path.append((segment_copy_number, segment_start_index, segment_end_index))
            segment_start_index = segment_end_index + 1
            segment_end_index = segment_start_index
            segment_copy_number = 2
            try:
                segments_rec = next(segments_iter)
            except StopIteration:
                segments_rec = None
                segments_iter = None
    return path


def try_getting_info_attribute(record,
                               attribute: str) -> int:
    try:
        value = record.INFO[attribute]
    except AttributeError:
        print('No {} field for record at position:{}'.format(attribute, record.POS))
    else:
        return value


def try_getting_format_attribute(record,
                                 sample_name: str,
                                 attribute: str) -> int:
    try:
        value = record.genotype(sample_name)[attribute]
    except AttributeError:
        print('No {} field for {} intervals at position:{}'.format(attribute, sample_name, record.POS))
    else:
        return value
