# generate_plot_data.py
#
# This script generates mock data for the QGP light-ion visualizations.
# The data is qualitatively modeled on the physical phenomena described in the text,
# such as nuclear modification and anisotropic flow, using simple mathematical functions.
# The output is a set of .dat files suitable for consumption by pgfplots.

import argparse
import os

import numpy as np


def generate_RAA_data(output_dir):
    """
    Generates mock data for the R_AA vs. p_T plot for both Pb-Pb and O-O collisions.
    """
    pT_pbpb = np.array([1, 2, 4, 7, 10, 15, 20])
    raa_pbpb = 0.8 * (1 - 0.7 * np.exp(-((pT_pbpb - 7) ** 2) / 15)) + 0.1
    err_pbpb = np.array([0.06, 0.05, 0.04, 0.03, 0.03, 0.04, 0.05])

    pT_oo = np.array([1, 2, 4, 6, 8, 10, 15, 20])
    dip = 0.35 * np.exp(-((pT_oo - 6) ** 2) / 10)
    peak = 0.07 * np.exp(-((pT_oo - 2) ** 2) / 2)
    raa_oo = 1.0 - dip + peak
    err_oo = np.array([0.08, 0.07, 0.06, 0.05, 0.05, 0.06, 0.07, 0.08])

    # Define file paths using the specified output directory.
    path_pbpb = os.path.join(output_dir, "RAA_PbPb.dat")
    path_oo = os.path.join(output_dir, "RAA_OO.dat")

    np.savetxt(path_pbpb, np.c_[pT_pbpb, raa_pbpb, err_pbpb], header="pT R_AA err_y", comments="# ")
    np.savetxt(path_oo, np.c_[pT_oo, raa_oo, err_oo], header="pT R_AA err_y", comments="# ")
    print(f"Generated data files: {path_pbpb}, {path_oo}")


def generate_flow_data(output_dir):
    """
    Generates mock data for the v_n vs. Centrality plot.
    """
    centrality = np.array([0, 5, 10, 20, 30, 40, 50, 60, 70])
    v2_oo = 0.068 - 0.00005 * (centrality - 40) ** 2
    v3_oo = 0.028 - 0.0002 * centrality
    v2_nene = v2_oo + 0.015 * np.exp(-centrality / 20)

    # Define file paths.
    path_v2_oo = os.path.join(output_dir, "flow_v2_OO.dat")
    path_v3_oo = os.path.join(output_dir, "flow_v3_OO.dat")
    path_v2_nene = os.path.join(output_dir, "flow_v2_NeNe.dat")

    np.savetxt(path_v2_oo, np.c_[centrality, v2_oo], header="centrality v2", comments="# ")
    np.savetxt(path_v3_oo, np.c_[centrality, v3_oo], header="centrality v3", comments="# ")
    np.savetxt(path_v2_nene, np.c_[centrality, v2_nene], header="centrality v2_NeNe", comments="# ")
    print(f"Generated data files: {path_v2_oo}, {path_v3_oo}, {path_v2_nene}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate mock data for QGP visualizations.")
    parser.add_argument(
        "--output-dir", type=str, default=".", help="Directory to save the .dat files"
    )
    args = parser.parse_args()

    # Create the output directory if it doesn't exist.
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Executing data generation for QGP visualizations in '{args.output_dir}'...")
    generate_RAA_data(args.output_dir)
    generate_flow_data(args.output_dir)
    print("Data generation complete.")
