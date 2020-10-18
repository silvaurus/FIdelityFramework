##########################################################
# Compute the FIT rate for target network
# Require: Input file that includes 
# (1) the size of each flip-flop category
# (2) the average activeness of flip-flop categories
# (3) the average 1 - Prob_SW_mask rate for every flip-flop categories
##########################################################
import argparse
import pandas as pd
import numpy as np

def get_fit_rate(args):
    ff_size, activeness, sw_nomask = pd.read_csv(args.input_file, sep=' ').to_numpy()

    fit_dp = np.sum(np.multiply(ff_size[:5], np.multiply(activeness[:5], sw_nomask[:5])))
    fit_local_ctrl = ff_size[5] * np.multiply(activeness[5], sw_nomask[5])
    fit_global_ctrl = ff_size[6] * np.multiply(activeness[6], sw_nomask[6])
    fit_sum = fit_dp + fit_local_ctrl + fit_global_ctrl
    return np.array([fit_dp, fit_local_ctrl, fit_global_ctrl, fit_sum]) * args.fit_raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fit_raw', type=float, default=600, help="The raw FIT rate")
    parser.add_argument('--input_file', required=True, help="The input file")

    args = parser.parse_args()

    out = get_fit_rate(args)
    print("+++++++++++++++++++++++++++++++++")
    print("Accelerator FIT Rate: {}".format(out[3]))
    print("Breakdown:") 
    print("Datapath: {}".format(out[0]))
    print("Local control: {}".format(out[1]))
    print("Global control: {}".format(out[2]))


if __name__ == '__main__':
    main()
