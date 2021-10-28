import argparse
import os
import numpy as np 

parser = argparse.ArgumentParser()
parser.add_argument('--score_files', nargs='+', type=str)
parser.add_argument('--score_weights', nargs='+', type=float, default=None)
parser.add_argument('--crop_agg', type=str, choices=['max', 'mean'], default='mean')


args = parser.parse_args()

score_npz_files = [np.load(x, allow_pickle=True) for x in args.score_files]

print(len(score_npz_files))

print(score_npz_files)

# if args.score_weights is None:
#     score_weights = [1] * len(score_npz_files)
# else:
#     score_weights = args.score_weights
#     if len(score_weights) != len(score_npz_files):
#         raise ValueError("Only {} weight specifed for a total of {} score files"
#                          .format(len(score_weights), len(score_npz_files)))

# score_list = [x['scores'][:, 0] for x in score_npz_files]
# label_list = [x['labels'] for x in score_npz_files]