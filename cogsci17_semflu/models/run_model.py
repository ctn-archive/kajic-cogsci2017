from wta_semflu import SemFlu
import numpy as np
import os
import pandas.rpy.common as com

from process_output import process_output

# Model parameters
d = 256                                          # dimensionality of vectors
sim_len = 20                                    # simulation length
amat = 'fan_mat'                             # association database to use
seed_start = 0
nr_seeds = 141                                   # number of simulations
wta_th = 0.25
nr_resp = 36                                   # nr of responses to process

fname = '{}_{}r_{}d_{}th_{}n_2bgr'.format(
    amat, nr_seeds, d, wta_th, nr_resp)  # dir-name to store simulations

seeds = np.arange(seed_start, nr_seeds)

base_dir = os.path.dirname(__file__)
results_dir = os.path.join(base_dir, 'data', fname)

for seed in seeds:
    SemFlu().run(
        d=d,
        seed=seed,
        sim_len=sim_len,
        wta_th=wta_th,
        amat=amat,
        data_dir=results_dir,
        backend='nengo_ocl')

print('Post-processing responses for R-analysis...')
output = process_output(results_dir, nr_resp)

csv_path = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, 'model_outputs',
    fname + '.csv')

r_df = com.convert_to_r_dataframe(output)
r_df.to_csvfile(csv_path)

print('Done with:', fname)
