#!/usr/bin/env python3

import smurf
import matplotlib.pyplot as plt
import vortector
import vortector.visualize
import argparse
import numpy as np
from simdata_vorticity import provide_simulation_data

import matplotlib as mpl
mpl.use('qt5agg')


def main():
    args = parse_cli_args()
    simid = args.simulation
    Noutput = args.Noutput

    levels = [float(x) for x in np.arange(-1, 1.5, 0.05)]

    Xc, Yc, A, vortensity, _, Rho, _ = provide_simulation_data(simid, Noutput)

    if args.rlim is not None:
        Rlims = args.rlim
    else:
        Rlims = [5.2, 15]
    nl = np.argmin(np.abs(Xc[:, 0]-Rlims[0]))
    nr = np.argmin(np.abs(Xc[:, 0]-Rlims[1]))
    vd = vortector.Vortector(Xc[nl:nr, :], Yc[nl:nr, :], A[nl:nr, :], vortensity[nl:nr, :], Rho[nl:nr, :],
                             verbose=False, med=0.15, mear=np.inf,
                             levels=levels
                             )

    vd.detect(include_mask=True)

    vortector.visualize.show_fit_overview_2D(vd)
    fig = plt.gca().get_figure()

    name = smurf.search.search(simid)[0]["name"]
    fig.suptitle(f"{simid} | {name} | N = {Noutput}")
    plt.show()


def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("simulation", type=str, help="Simulation ID.")
    parser.add_argument("Noutput", type=int, help="Output number.")
    parser.add_argument("--rlim", type=float, nargs=2, help="Radial limits.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
