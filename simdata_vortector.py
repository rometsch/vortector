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

    levels = [float(x) for x in np.arange(-1, 1, 0.05)]

    R, Phi, A, vorticity, vorticity_Kepler, Rho, Rho_background = provide_simulation_data(
        simid, Noutput)

    if args.vorticity:
        detection_quantity = vorticity/vorticity_Kepler
    else:
        vortensity = vorticity/vorticity_Kepler * Rho_background/Rho
        detection_quantity = vortensity

    if args.rlim is not None:
        Rlims = args.rlim
    else:
        Rlims = [5.2, 15]
    nl = np.argmin(np.abs(R[:, 0]-Rlims[0]))
    nr = np.argmin(np.abs(R[:, 0]-Rlims[1]))
    vd = vortector.Vortector(R[nl:nr, :], Phi[nl:nr, :], A[nl:nr, :], detection_quantity[nl:nr, :], Rho[nl:nr, :],
                             verbose=False, med=0.15, mear=np.inf,
                             levels=levels
                             )

    vd.detect(include_mask=True)

    vortector.visualize.show_fit_overview_2D(vd)
    fig = plt.gca().get_figure()

    if args.vorticity:
        for ax in fig.axes:
            if ax.get_xlabel() == r"$\varpi/\varpi_0$":
                ax.set_xlabel(r"$(\nabla \times \vec{v})_z \,/\, 0.5 \Omega_\mathrm{K}$")

    name = smurf.search.search(simid)[0]["name"]
    fig.suptitle(f"{simid} | {name} | N = {Noutput}")
    plt.show()


def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("simulation", type=str, help="Simulation ID.")
    parser.add_argument("Noutput", type=int, help="Output number.")
    parser.add_argument("--rlim", type=float, nargs=2, help="Radial limits.")
    parser.add_argument("--vorticity", action="store_true", help="Use vorticity instead of vortensity.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
