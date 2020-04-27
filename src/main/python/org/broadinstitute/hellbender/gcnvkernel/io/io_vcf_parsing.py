import logging
import numpy as np
import pandas as pd
import pyvcf
from typing import TypeVar, List, Tuple

_logger = logging.getLogger(__name__)

def read_sample_segments_and_calls(clustered_vcf: str,
                                 sample_name: str,
                                 contig: str,
                                 contig_interval_list: str) -> List[Tuple[TypeVar('_T'), int, int]]:
    segments = vcf.Reader(filename = clustered_vcf)