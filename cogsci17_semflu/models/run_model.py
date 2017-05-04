from wta_semflu import SemFlu
import numpy as np
import os
import argparse
import pandas.rpy.common as com

from process_output import process_output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'database', nargs=1, type=str, help="Source of word database" +
        "(ngram, fan_mat, beagle)")
    parser.add_argument(
        'th', nargs=1, type=float, help="WTA threshold (0.25 for fan_mat)",
        default=0.3)
    args = parser.parse_args()

    amat = args.database[0]
    wta_th = args.th[0]

    # Model parameters
    d = 256                         # dimensionality of vectors
    sim_len = 20                    # simulation length
    seed_start = 0
    nr_seeds = 141                  # number of simulations
    nr_resp = 36                    # nr of responses to process

    # dir-name to store simulations
    fname = '{}_{}r_{}d_{}th_{}n_157w'.format(
            amat, nr_seeds, d, wta_th, nr_resp)
    print amat, wta_th, fname

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

    print 'Post-processing responses for R-analysis...'
    output = process_output(results_dir, nr_resp)

    csv_path = os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir, 'model_outputs',
        fname + '.csv')

    r_df = com.convert_to_r_dataframe(output)
    r_df.to_csvfile(csv_path)

    print 'Done with:', fname
