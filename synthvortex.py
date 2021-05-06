import os
import numpy as np
from numpy.random import default_rng

try:
    from opensimplex import OpenSimplex
except ImportError:
    print("'opensimplex' package not found. Disable noise. Run 'python3 -m pip install opensimplex'")
    OpenSimplex = None


class VortexGenerator:

    def __init__(self, Nvortices, Nr=121, Nphi=251, rmin=1, rmax=20, phimin=-np.pi, phimax=np.pi, c_dens=1,
                 a_dens=3, a_vort=-1.5, noise=0.05, seed=None, periodic_x=False, periodic_y=False):
        self.Nr = Nr
        self.Nphi = Nphi
        self.rmin = rmin
        self.rmax = rmax
        self.phimin = phimin
        self.phimax = phimax
        self.Lr = rmax - rmin
        self.Lphi = phimax - phimin
        self.seed = seed
        self.c_dens = c_dens
        self.a_dens = a_dens
        self.a_vort = a_vort

        self.noise = noise
        if seed is None:
            seed = int.from_bytes(os.urandom(16), 'big')
        self.seed = seed

        self.periodic_x = periodic_x
        self.periodic_y = periodic_y

        self.generate_grid()

        self.vortex_params = []

        for n in range(Nvortices):
            self.add_vortex(seed=seed+n)

        self.calc_data()

    def calc_data(self):
        vorticity = np.ones((self.Nr, self.Nphi))
        density = self.c_dens*np.ones((self.Nr, self.Nphi))

        for r0, sigma_r, phi0, sigma_phi, a_vort, a_dens in self.vortex_params:
            ex = np.exp(- 0.5*(self.Rs-r0)**2/sigma_r**2)
            ey = np.exp(- 0.5*(self.Phis-phi0)**2/sigma_phi**2)
            profile = ex*ey
            vorticity += a_vort*profile
            density += a_dens*profile
            if self.periodic_x:
                Dx = self.Lr if r0 < 0.5*(self.rmin + self.rmax) else -self.Lr
                ex_per = np.exp(- 0.5*(self.Rs - (r0+Dx))**2/sigma_r**2)
                profile = ex_per*ey
                vorticity += a_vort*profile
                density += a_dens*profile
            if self.periodic_y:
                Dy = self.Lphi if phi0 < 0.5 * \
                    (self.phimin + self.phimax) else -self.Lphi
                ey_per = np.exp(- 0.5*(self.Phis - (phi0+Dy))**2/sigma_phi**2)
                profile = ex*ey_per
                vorticity += a_vort*profile
                density += a_dens*profile
            if self.periodic_x and self.periodic_y:
                profile = ex_per*ey_per
                vorticity += a_vort*profile
                density += a_dens*profile

        if self.noise > 0:
            vorticity += self.noise*self.a_vort * \
                simplex_noise(self.Nr, self.Nphi, seed=self.seed)
            density += self.noise*self.c_dens * \
                simplex_noise(self.Nr, self.Nphi, seed=self.seed**2)

        vortensity = vorticity/density

        self.vortensity = vortensity
        self.vorticity = vorticity
        self.density = density

    def add_vortex(self, r0=None, sigma_r=None, phi0=None, sigma_phi=None,
                   a_vort=None, a_dens=None, seed=None):
        rng = default_rng(seed)
        if r0 is None:
            r0 = rng.uniform(low=self.rmin, high=self.rmax)
        if sigma_r is None:
            sigma_r = rng.uniform(low=self.rmin/20, high=self.rmax/4)
        if phi0 is None:
            phi0 = rng.uniform(low=-np.pi, high=np.pi)
        if sigma_phi is None:
            sigma_phi = rng.uniform(low=np.pi/20, high=np.pi/4)
        if a_vort is None:
            a_vort = rng.random()*self.a_vort
        if a_dens is None:
            a_dens = rng.random()*self.a_dens
        if seed is None:
            seed = 0

        self.vortex_params.append(
            [r0, sigma_r, phi0, sigma_phi, a_vort, a_dens]
        )

    def generate_grid(self):
        # grid coordinates, use squared cells
        ris = np.geomspace(self.rmin, self.rmax, self.Nr+1)  # cell interfaces
        phiis = np.linspace(self.phimin, self.phimax,
                            self.Nphi+1)  # cell interfaces

        # calculate cell centers, sizes
        drs = ris[1:] - ris[:-1]
        rs = 0.5*(ris[1:] + ris[:-1])
        dphis = phiis[1:] - phiis[:-1]
        phis = 0.5*(phiis[1:] + phiis[:-1])

        # calculate cell area along the radial direction
        area = dphis[0] * 0.5 * (ris[1:]**2 - ris[:-1]**2)

        # create meshgrids
        Phis, Rs = np.meshgrid(phis, rs)

        # calculate the area of the cells
        Area = np.tile(area, (Rs.shape[1], 1)).T

        self.ris = ris
        self.phiis = phiis
        self.drs = drs
        self.rs = rs
        self.dphis = dphis
        self.phis = phis
        self.area = area
        self.Rs = Rs
        self.Phis = Phis
        self.Area = Area


if OpenSimplex is not None:
    def simplex_noise(Nx, Ny, seed=1234, order=3, feature_size=None):
        if feature_size is None:
            feature_size = max(Nx, Ny)/13
        ng = OpenSimplex(seed)
        noise = np.zeros((Nx, Ny))
        for norder in range(1, order+1):
            feature_size /= 2
            for nx in range(Nx):
                for ny in range(Ny):
                    noise[nx, ny] = (1/2**norder) * \
                        ng.noise2d(nx/feature_size, ny/feature_size)
            return noise
else:
    def simplex_noise(Nx, Ny, seed=0, order=3, feature_size=None):
        return np.zeros((Nx, Ny))
