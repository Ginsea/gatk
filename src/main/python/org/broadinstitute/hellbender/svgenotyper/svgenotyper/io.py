import numpy as np
import pandas as pd
import torch

from gatktool import tool

from . import constants
from .constants import SVTypes
from .preprocess import compute_preprocessed_tensors
from .model import SVGenotyperData


def load_batch(batch_size: int,
               device: str,
               svtype: SVTypes,
               tensor_dtype: torch.dtype):
    vid_list = []
    svlen_list = []
    pe_list = []
    sr1_list = []
    sr2_list = []
    ncn_list = []
    cnlp_list = []
    for _ in range(batch_size):
        fifo_line = tool.readDataFIFO()
        fifo_data = fifo_line.split('\t')
        vid_list.append(fifo_data[0])
        svlen_list.append(int(fifo_data[1]))
        pe_list.append([int(x) for x in fifo_data[2].split(';')])
        sr1_list.append([int(x) for x in fifo_data[3].split(';')])
        sr2_list.append([int(x) for x in fifo_data[4].split(';')])
        ncn_list.append([int(x) for x in fifo_data[5].split(';')])
        if svtype == SVTypes.DEL or svtype == SVTypes.DUP:
            cnlp_list.append([[int(y) for y in x.split(',')] for x in fifo_data[6].split(';')])
    vid_np = np.asarray(vid_list)
    svlen_t = torch.tensor(svlen_list, device=device, dtype=tensor_dtype)
    pe_t = torch.tensor(pe_list, device=device, dtype=tensor_dtype)
    sr1_t = torch.tensor(sr1_list, device=device, dtype=tensor_dtype)
    sr2_t = torch.tensor(sr2_list, device=device, dtype=tensor_dtype)
    ncn_t = torch.tensor(ncn_list, device=device, dtype=tensor_dtype)
    cnlp_t = torch.tensor(cnlp_list, device=device, dtype=tensor_dtype)
    return vid_np, pe_t, sr1_t, sr2_t, ncn_t, svlen_t, cnlp_t


def load_data(batch_size: int,
              mean_coverage_path: str,
              samples_path: str,
              svtype: SVTypes,
              num_states: int,
              depth_dilution_factor: float,
              tensor_dtype: torch.dtype,
              device: str = 'cpu'):
    # TODO: cross-check depth table samples with samples list
    mean_count_df = pd.read_csv(mean_coverage_path, sep='\t', header=None, index_col=0)
    mean_count_t = torch.from_numpy(mean_count_df.values).to(device=device, dtype=tensor_dtype).squeeze(-1) / torch.tensor(constants.DEPTH_PLOIDY).to(device=device, dtype=tensor_dtype)
    samples_np = np.loadtxt(samples_path, dtype=str)
    vids_np, pe_t, sr1_t, sr2_t, ncn_t, svlen_t, cnlp_t = load_batch(batch_size=batch_size, device=device, svtype=svtype, tensor_dtype=tensor_dtype)
    if vids_np.shape[0] == 0:
        return None

    pe_t, sr1_t, sr2_t, depth_t, rd_gt_prob_t = compute_preprocessed_tensors(num_states, svtype, depth_dilution_factor,
                                                                             pe_t, sr1_t, sr2_t, mean_count_t, cnlp_t,
                                                                             ncn_t, device, tensor_dtype=tensor_dtype)
    return SVGenotyperData(svtype, vids_np, samples_np, pe_t, sr1_t, sr2_t, depth_t, svlen_t, rd_gt_prob_t)


def write_variant_output(output_path: str, output_data: dict):
    param_keys = ["p_m_pe", "p_m_sr1", "p_m_sr2",  "eps_pe", "eps_sr1", "eps_sr2",
                  "phi_pe", "phi_sr1", "phi_sr2"]
    header = ["vid", "freq_z"] + param_keys
    with open(output_path, 'w') as f:
        line = "#" + "\t".join(header)
        f.write(line + "\n")
        for vid, dat in output_data.items():
            z_freq = pretty_print_2d_array(dat['freq_z'])
            line = "\t".join([vid, z_freq] + [str(dat[x]) for x in param_keys])
            f.write(line + "\n")


def pretty_print_2d_array(arr):
    return ";".join(",".join(str(y) for y in x) for x in arr.tolist())


def save_tensors(data: SVGenotyperData, base_path: str):
    data_vars = vars(data)
    for var in data_vars:
        if data_vars[var] is not None:
            torch.save(data_vars[var], base_path + "." + var + ".pt")


def save_list(data: list, path: str):
    with open(path, 'w') as f:
        f.writelines([x + '\n' for x in data])


def load_tensors(base_path: str, svtype: SVTypes, tensor_dtype: torch.dtype, device: str = 'cpu'):
    pe_t = torch.load(base_path + ".pe_t.pt", map_location=device).to(dtype=tensor_dtype)
    sr1_t = torch.load(base_path + ".sr1_t.pt", map_location=device).to(dtype=tensor_dtype)
    sr2_t = torch.load(base_path + ".sr2_t.pt", map_location=device).to(dtype=tensor_dtype)
    depth_t = torch.load(base_path + ".depth_t.pt", map_location=device).to(dtype=tensor_dtype)
    svlen_t = torch.load(base_path + ".svlen_t.pt", map_location=device).to(dtype=tensor_dtype)
    if svtype == SVTypes.DEL or svtype == SVTypes.DUP:
        rd_gt_prob_t = torch.load(base_path + ".rd_gt_prob_t.pt", map_location=device).to(dtype=tensor_dtype)
    else:
        rd_gt_prob_t = None
    vids = np.loadtxt(base_path + ".vids.list", dtype=str)
    samples = np.loadtxt(base_path + ".sample_ids.list", dtype=str)
    return SVGenotyperData(svtype=svtype, vids=vids, samples=samples, pe_t=pe_t, sr1_t=sr1_t, sr2_t=sr2_t,
                           depth_t=depth_t, svlen_t=svlen_t, rd_gt_prob_t=rd_gt_prob_t)


def load_list(path: str):
    with open(path, 'r') as f:
        lines = f.readlines()
        if lines is None or len(lines) == 0:
            raise ValueError("Tried to read empty list: {:s}".format(path))
        return [x.strip() for x in lines]
