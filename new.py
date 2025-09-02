import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy.optimize import curve_fit
from tqdm import tqdm

kB = const.Boltzmann
q = const.e
h = const.h
c = const.c
PulsePower1st = 1
k=8.99e9

n0 = 5e17
# D0 = 0.26  # to be confirmed
k1 = 14e7
k2 = 5e-9
k3 = 1e-30
L = 1e-3
a = 0  # D modifier: not doing a lot at the moment
mb = 100

npu0 = 70e17
npr0 = 5e17
T = 10

t0 = 3e-11
t_max = 1.25e-8
dt = 1e-12
dx = 0.5e-5
xpu = 5e-4
xpr = 5e-4
w = 2e-4
JP = 1

voltage_range = range(4, 8, 1)

x_range = np.arange(0, L + dx, dx)
dnx = np.zeros(len(x_range))
n_initial = np.zeros(len(x_range)) + n0
Nt = int(t_max / dt)

# Lists to store Q_trapz values and td values
td_values = np.arange(-0.5e-9, 0.51e-9, 0.05e-9)


def D(n):
    return mb * kB * T / q * (n0 ** a) / np.power(n, a)


def diffusion(n):
    return D(n) * (np.roll(n, -1) - np.roll(n, 1)) / (2 * dx)


def drift(n, E):
    # return (n-n0) * mb * E
    return n * mb * E


def gaussian(x, x_0, w=1e-4):
    return np.exp(-1 * np.square((x - x_0) / w))


def expo(x, a, b, t):
    return a + b * np.exp(x / t)

def compute_total_field(n):
    qn = n*q
    qp = n[::-1]*-(q)
    Q=qn+qp
    print (n-n[::-1])


    Ef = np.zeros(len(x_range))
    for j in range(len(x_range)):
        E_j = 0
        for i in range(len(n)):
            r = (j - i) * dx
            if r != 0:
                E_j += k * Q[i] / (r * abs(r))
        Ef[j] = E_j
    return Ef


# Loop over voltage values and td values to calculate Q_avg - Q_avg0
for V in voltage_range:
    E = V / L
    n = n_initial.copy()
    n1 = npu0 * gaussian(x_range, xpu)
    n += n1
    n[0] = n0
    n[-1] = n0

    # Loop over different td values
    loop = 0
    for td in tqdm(td_values, desc=f"Processing td values for V={V}", colour="green"):
        iteration = 1
        n = n_initial.copy()
        n1 = npu0 * gaussian(x_range, xpu)
        n2 = npr0 * gaussian(x_range, xpr)
        if td > 0:
            n1st = n1
            n2nd = n2
        else:
            n1st = n2
            n2nd = n1
        n += n1st
        for t in range(Nt):
            if t == int(abs(td) / dt):
                n += n2nd
            Ef=compute_total_field(n)
            Etotal=E+Ef
            nprime = (np.roll(n, -1) - np.roll(n, 1)) / (2 * dx)
            dif = (np.roll(diffusion(n), -1) - np.roll(diffusion(n), 1)) / (2 * dx)
            driE = (np.roll(drift(n, E), -1) - np.roll(drift(n, E), 0)) / dx
            # rec = k1 * (n - n0) + k2 * (n ** 2 - n0 ** 2)
            rec = k1 * (n - n0) + k2 * n * (n - n0) + k3 * n * n * (n - n0)

            dnx = (dif + driE - rec) * dt
            n += dnx

            n[0] = n0
            n[-1] = n0
            p = n[::-1]
            iteration += 1


            if t%50 == 0:
               plt.plot(x_range,n)
               plt.plot(x_range, p)
               plt.plot(x_range, Etotal)
               #plt.ylim(0.4e18, 1e18)
               plt.show()