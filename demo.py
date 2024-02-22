#!/usr/bin/env python3

import os
import vortector
import vortector.visualize
import configparser
import matplotlib.pyplot as plt
import argparse
import numpy as np

import matplotlib as mpl
mpl.use('qt5agg')


def read_2D_file(filename, Nx, Ny):
    with open(filename, "rb") as in_file:
        rv = np.fromfile(in_file)
    return rv.reshape(Nx, Ny)


def main():
    args = parse_cli_args()

    datadir = os.path.dirname(args.config_file)
    config = configparser.ConfigParser()
    config.read(args.config_file)
    Nr = int(config["data"]["Nr"])
    Nphi = int(config["data"]["Nphi"])

    r = read_2D_file(
        os.path.join(datadir, config["data"]["r"]),
        Nr, Nphi)
    phi = read_2D_file(
        os.path.join(datadir, config["data"]["phi"]),
        Nr, Nphi)
    area = read_2D_file(
        os.path.join(datadir, config["data"]["area"]),
        Nr, Nphi)
    vortensity = read_2D_file(
        os.path.join(datadir, config["data"]["vortensity"]),
        Nr, Nphi)
    density = read_2D_file(
        os.path.join(datadir, config["data"]["density"]),
        Nr, Nphi)

    if "rmin" in config["parameters"]:
        rmin = float(config["parameters"]["rmin"])
        nl = np.argmin(np.abs(r[:,0]-rmin))
    else:
        nl = 0
        
    if "rmin" in config["parameters"]:
        rmax = float(config["parameters"]["rmax"])
        nr = np.argmin(np.abs(r[:,0]-rmax))
    else:
        nr = 0
    
    vd = vortector.Vortector(r, phi, area, vortensity, density)

    vd.detect(include_mask=True)

    vortector.visualize.show_fit_overview_2D(vd)
    fig = plt.gca().get_figure()

    plt.show()


def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Config file.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
