import argparse
import numpy as np
from Bio import SeqIO
import pandas as pd
from collections import defaultdict
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluating the generation performance')
    parser.add_argument('--hi_form_path', default="", type=str)
    parser.add_argument('--sequence_file', default="", type=str)
    parser.add_argument('--index_pair', default="", type=str)
    parser.add_argument('--ground_truth_path', default="", type=str)
    args = parser.parse_args()
    return args

def read_ground_truth(path):
    sum_of_freq = 0.0
    seq2freq = defaultdict(float)
    for record in SeqIO.parse(path, "fasta"):
        descs = record.description.split("|")
        descs = {x.split("=")[0]: x.split("=")[1] for x in descs}
        seq2freq[str(record.seq)] += float(descs["freq"])
        sum_of_freq += float(descs["freq"])
    # print(sum_of_freq)
    # exit()
    return seq2freq

def read_fasta(path):
    seq2strain_name = defaultdict(set)
    accid2seq = {}
    for record in SeqIO.parse(path, "fasta"):
        descs = record.description.split("|")
        # find the EPI accession (not the EPI_ISL one) anywhere in the header
        accid = None
        for x in descs:
            if x.startswith("EPI") and "EPI_ISL" not in x:
                accid = x
                break
        if accid is None:
            continue
        # find the strain name (the field starting with A/ or B/)
        strain = None
        for x in descs:
            if x.startswith("A/") or x.startswith("B/"):
                strain = x
                break
        accid2seq[accid] = str(record.seq)
        if strain is not None:
            seq2strain_name[str(record.seq)].add(strain)
    return accid2seq, seq2strain_name

def calculate_hi_matrix(df_hi, clade_dict, vaccine_dict, virus2clade):
    hi_matrix = np.zeros((len(clade_dict), len(vaccine_dict)))
    hi_matrix_mask = np.zeros((len(clade_dict), len(vaccine_dict)))

    for virus, reference, hi in zip(df_hi["virus"], df_hi["reference"], df_hi["hi"]):
        if virus not in virus2clade or not virus2clade[virus]:
            continue
        i = clade_dict[virus2clade[virus]]
        j = vaccine_dict[reference]
        hi_matrix[i, j] += hi # Taking the log
        hi_matrix_mask[i, j] += 1

    hi_matrix = hi_matrix / (hi_matrix_mask + 1e-20) # Taking the average over log HI value
    hi_matrix[hi_matrix_mask == 0] = 0
    return hi_matrix, hi_matrix_mask

def get_score(clade2freq, clade_set, hi_fold, hi_mask):
    # clade_set: clade set that has HI data
    # hi_mask & hi_fold: [num_virus, num_vaccines]
    freq = np.asarray([clade2freq.get(clade, 0.0) for clade in clade_set])
    freq = freq.reshape(hi_fold.shape[0], 1) # [C, 1]
    norm_freq = freq * (hi_mask > 0) / np.sum(freq * (hi_mask > 0) + 1e-20, axis=0, keepdims=True)
    vaccine_score = np.sum(hi_fold * norm_freq, axis=0) #  * (hi_matrix_mask > 0), axis=0) / np.sum(gt_score * (hi_matrix_mask > 0), axis=0)
    vaccine_score_mask = (np.sum(norm_freq, axis=0) > 0)
    return freq, vaccine_score, vaccine_score_mask

if __name__ == "__main__":
    args = parse_args()

    # read all HI pairs (all experiments)
    df_hi = pd.read_csv(args.hi_form_path)
    virus_list = list(set(df_hi["virus"]))
    vaccine_list = list(set(df_hi["reference"]))
    vaccine_dict = {x: i for i, x in enumerate(vaccine_list)}
    print("# of vaccines", len(vaccine_list))

    # Read accid and sequences
    accid2seq, seq2strain_names = read_fasta(args.sequence_file)

    # Get the clade/cluster for each sequences
    virus2seq = {}
    seq2viruses = defaultdict(list)

    for virus in virus_list:
        virus_seq = accid2seq[virus]
        seq2viruses[virus_seq].append(virus)
        virus2seq[virus] = accid2seq[virus]
                
    seq_set = list(seq2viruses.keys())
    seq_dict = {x: i for i, x in enumerate(seq_set)}

    print("Seq", len(seq_set))

    hi_fold_seq, hi_matrix_mask_seq = calculate_hi_matrix(df_hi, seq_dict, vaccine_dict, virus2seq)

    # Read ground-truth scores (seq level, cluster level and clade level)
    # - Get frequecies
    gt_seq2freq = read_ground_truth(args.ground_truth_path)
    # - Get scores
    gt_seq_freqs, gt_seq_score, gt_seq_score_mask = get_score(gt_seq2freq, seq_set, hi_fold_seq, hi_matrix_mask_seq)
    vaccine_coverage_seq = np.sum((hi_matrix_mask_seq > 0) * gt_seq_freqs, axis=0)
    
    vaccine_seq2score_seq_level = defaultdict(list)
    vaccine_seq2coverage_seq_level = defaultdict(list)
    for i, v in enumerate(vaccine_list):
        seq = accid2seq[v]
        vaccine_seq2score_seq_level[seq].append(gt_seq_score[i])
        vaccine_seq2coverage_seq_level[seq].append(vaccine_coverage_seq[i])

    # taking averge, if some vaccines sharing the same sequences
    for key in vaccine_seq2score_seq_level:
        vaccine_seq2score_seq_level[key] = sum(vaccine_seq2score_seq_level[key]) / len(vaccine_seq2score_seq_level[key])
    
    for key in vaccine_seq2coverage_seq_level:
        vaccine_seq2coverage_seq_level[key] = sum(vaccine_seq2coverage_seq_level[key]) / len(vaccine_seq2coverage_seq_level[key])
    
    df = pd.read_csv(args.index_pair)
    scores_seq = []
    coverage_seq = []
    for vaccine_id in df["reference"]:
        vaccine_seq = accid2seq[vaccine_id]
        scores_seq.append(vaccine_seq2score_seq_level.get(vaccine_seq, None))
        coverage_seq.append(vaccine_seq2coverage_seq_level.get(vaccine_seq, None))

    df["gt_score_seq"] = scores_seq
    df["coverage_seq"] = coverage_seq
    df["strain_name"] = [ "|".join(list(seq2strain_names[accid2seq[accid]]))  for accid in df["reference"]]

    df = df.drop(columns='reference_seq')

    print(args.index_pair.split(".csv")[0] + "_and_gt.csv")
    df.to_csv(args.index_pair.split(".csv")[0] + "_and_gt.csv", index=False)
