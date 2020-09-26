#!/usr/bin/python
"""
Adapted from:

Lawrence K Forbes1 and Stuart Crozier (2001)
A novel target-field method for finite-length magnetic resonance shim coils:
    I. Zonal shims
doi: 10.1088/0022-3727/34/24/305

"""
import numpy as np
import scipy.integrate
import scipy.interpolate
from scipy.special import ellipk
from scipy.special import ellipe
from sympy import KroneckerDelta

class ZonalShimCylindrical1D:
    n_trapz = 50
    processes = 7
    H_max = 1
    lam = 10**(-16)
    N = 20  ## K = N
    N_coil_pos = 20
    N_plot_points = 200

    L = None
    a = None
    c = None
    p = None
    q = None

    S_array = None
    R_array = None
    P_array = None

    coils_positive = list()
    coils_negative = list()

    def __init__(self, L, a, c, p, q):
        self.L = L # Location of the cylinder from -L < z < L
        self.a = a # radius of the cylinder
        self.c = c # radius of the target region
        self.p = p # target region location from pL < z < qL
        self.q = q # -1 < p < q < 1

    def M(self, z, z_dash):
        """
        solution of target integral formula (14)
        ellipk/e package from scipy
        """
        return (1 / (np.pi * np.sqrt((self.a + self.c)**2 + (z_dash - z)**2))
                * ((self.c**2 - self.a**2 + (z_dash - z)**2) / \
                   ((self.a - self.c)**2 + (z_dash - z)**2)
                * ellipe(4 * self.a * self.c / ((self.a + self.c)**2 + (z_dash - z)**2 ))
                - ellipk(4 * self.a * self.c / ((self.a + self.c)**2 + (z_dash - z)**2 ))))

    def phi_n(self, z, n):
        """
        phi_n(z), formula (16) basis function which represents the sinulodila \
        approximation for the range where target field is defined.

        """
        return np.sin(n * np.pi * (z + self.L) / (2 * self.L))

    def T(self, z, n):
        """
        Equation (18), eg. convolution.
        Integration from -L to L using trapz rule
        """
        dz_dash_value = np.linspace(-self.L, self.L, self.n_trapz)
        dy_value = self.phi_n(dz_dash_value, n) * self.M(z, dz_dash_value)
        return scipy.integrate.trapz(dy_value, dz_dash_value)

    def H_c_target_linear(self, z):
        """
        Compute the Linear target field, according to equation (29) and (30).
        """
        return 2.0 * self.H_max / (self.q - self.p) * (z / self.L - (self.p + self.q)/2.0)

    def S_n_k(self, n, k):

        dz_value = np.linspace(self.L*self.p, self.L*self.q, self.n_trapz)
        dy_value = np.zeros_like(dz_value)
        for N in range(dz_value.size):
            dy_value[N] = self.T(dz_value[N], n) * self.T(dz_value[N], k)

        return scipy.integrate.trapz(dy_value, dz_value) \
                + KroneckerDelta(n, k) * (n**4 * np.pi**4 *self.lam) / (16.0 * self.L**3)

    def R_n_linear(self, n):
        # if value wrong -> correct here!
        dz_value = np.linspace(self.L * self.p, self.L * self.q, self.n_trapz)
        dy_value = np.zeros_like(dz_value)
        for N in range(dz_value.size):
            dy_value[N] = self.H_c_target_linear(dz_value[N]) * self.T(dz_value[N], n)
        return (-1) * scipy.integrate.trapz(dy_value, dz_value)

    def j_sum(self, z):
        j_array = np.zeros_like(self.P_array)
        for N in range( self.P_array.size ):
            j_array[N] = self.phi_n(z, N+1) * self.P_array[N]
        return np.sum(j_array)

    def stream_func_sum(self, z):
        sf_array = np.zeros_like(self.P_array)
        for N in range( self.P_array.size ):
            n = N+1
            sf_array[N] = ((2 * self.L)/(n * np.pi) * self.P_array[N]
                           * np.cos(n * np.pi * (z + self.L) / (2 * self.L)))
        return -1 * np.sum(sf_array)

    def calParam(self):
        """
        evaluate the parameters of S and R
        N x N matrix is symmetric ?!
        S(n,k) * P(k) = R(n)
        """
        self.S_array = np.zeros((self.N, self.N))
        self.R_array = np.zeros(self.N)

        for n in range(1, self.N + 1):
            self.R_array[n-1] = self.R_n_linear(n)
            for k in range(1, self.N+1):
                self.S_array[n-1][k-1] = self.S_n_k(n, k)
        self.P_array = scipy.linalg.solve( self.S_array, self.R_array)

    def plot_current_density(self, filename=None):
        z = np.linspace(-self.L, self.L, self.N_plot_points)
        j = np.zeros_like(z)

        for n in range( z.size ):
            j[n] = self.j_sum(z[n])
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax  = plt.axes()

        if filename is None:
            ax.plot(z, j)
            plt.show()

    def plotSt(self, plot=True, plot_coil_pos=True, filename=None):
        z  = np.linspace(-self.L, self.L, self.N_plot_points)
        j  = np.zeros_like(z)
        st_int = np.zeros_like(z)

        for n in range(z.size):
            j[n] = self.j_sum(z[n])
            st_int[n] = self.stream_func_sum(z[n])
        # st_int = scipy.integrate.cumtrapz(j, z, initial=0)

        import matplotlib.pyplot as plt
        fig, ax1_j = plt.subplots()
        ax2_int = ax1_j.twinx()
        ax1_j.plot( z, j, "b" )
        ax1_j.set_ylabel("current density j in A per metre", color="b")
        ax2_int.plot( z, st_int , "r--")
        ax2_int.set_ylabel("stream function of current density", color="r")

        if plot_coil_pos:
            self.cal_coil_positions(z, st_int, j)
            for n_roots in self.coils_negative:
                plt.axvline(n_roots, color="red")
            for n_roots in self.coils_positive:
                plt.axvline(n_roots, color="green")

        if plot is True and filename is None: # stub
            plt.show()

    def cal_coil_positions(self, z, st_int, j, plot=True, filename=None):
        centre_spacing = (st_int.max() - st_int.min()) / self.N_coil_pos
        coil_positions_y = np.linspace(st_int.min() + 0.5 * centre_spacing,
                                          st_int.max() - 0.5 * centre_spacing,
                                          self.N_coil_pos)

        self.coils_positive = []
        self.coils_negative = []

        # get the positions
        for n_coil_position in coil_positions_y:
            ## make spline approximation of the coil positions
            f = scipy.interpolate.UnivariateSpline(z, st_int - (n_coil_position), s=0)
            c_density_spline = scipy.interpolate.UnivariateSpline(z, j, s=0)
            roots = f.roots()

            for n_roots in roots:
                if c_density_spline(n_roots) > 0:
                    self.coils_positive.append(n_roots)
                elif c_density_spline(n_roots) < 0:
                    self.coils_negative.append(n_roots)
                if c_density_spline(n_roots) == 0:
                    print("warning, coil at 0.0")


if __name__ == "__main__":
    paper_example = ZonalShimCylindrical1D(0.2, 0.1, 0.05, -0.7, 0.1)
    paper_example.calParam()
    paper_example.plotSt()

    symetric_example = ZonalShimCylindrical1D(L=0.2, a=0.1, c=0.05, p=-0.2, q=0.2)
    symetric_example.calParam()
    symetric_example.plotSt()
