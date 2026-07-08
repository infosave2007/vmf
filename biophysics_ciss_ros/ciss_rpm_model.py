#!/usr/bin/env python3
"""
Radical-pair model of CISS-biased superoxide production in the
mitochondrial electron-transport chain. Liouville-von Neumann evolution
with Haberkorn spin-selective recombination. Produces fig_ros_vs_B.pdf
and the numeric yields quoted in the paper.

Spin system: two electron spins (A = flavin semiquinone, B = O2^-) and
one spin-1/2 nucleus hyperfine-coupled to A (superoxide is treated as
hyperfine-free). Hilbert space dim 8; Liouvillian 64x64.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Pauli / spin-1/2 operators
sx = 0.5*np.array([[0,1],[1,0]], complex)
sy = 0.5*np.array([[0,-1j],[1j,0]], complex)
sz = 0.5*np.array([[1,0],[0,-1]], complex)
I2 = np.eye(2, dtype=complex)

def kron(*a):
    out = a[0]
    for x in a[1:]:
        out = np.kron(out, x)
    return out

# order: electron A (x) electron B (x) nucleus
SAx,SAy,SAz = [kron(s,I2,I2) for s in (sx,sy,sz)]
SBx,SBy,SBz = [kron(I2,s,I2) for s in (sx,sy,sz)]
Ix ,Iy ,Iz  = [kron(I2,I2,s) for s in (sx,sy,sz)]

# singlet / triplet projectors on the electron pair
SASB = SAx@SBx + SAy@SBy + SAz@SBz
PS = 0.25*np.eye(8) - SASB           # singlet projector
PT = np.eye(8) - PS                  # triplet projector

gamma_e = 1.7609e8                   # rad/s per mT (g=2)
A_HF    = 1.0                        # mT, flavin effective hyperfine
kS, kT  = 1.0e7, 1.0e6              # s^-1, singlet/triplet recombination

def liouvillian(B_mT):
    w = gamma_e*B_mT
    H = w*(SAz+SBz) + gamma_e*A_HF*(SAx@Ix + SAy@Iy + SAz@Iz)
    d = 8
    Id = np.eye(d, dtype=complex)
    # dvec(rho)/dt = L vec(rho); L = -i(H⊗I - I⊗H^T) - 1/2 sum k(P⊗I + I⊗P^T)
    L = -1j*(np.kron(H, Id) - np.kron(Id, H.T))
    L += -0.5*kS*(np.kron(PS, Id) + np.kron(Id, PS.T))
    L += -0.5*kT*(np.kron(PT, Id) + np.kron(Id, PT.T))
    return L

def expm_herm(M):
    """exp(M) for anti-Hermitian M via eigendecomposition."""
    w, V = np.linalg.eig(M)
    return V @ np.diag(np.exp(w)) @ np.linalg.inv(V)

def yields(B_mT, P):
    """Triplet (superoxide) and singlet yields for CISS polarization P.

    Physical CISS initial state (Fay 2021; Luo & Hore 2021): the pair is
    born SINGLET (bonded electron transfer), and the CISS electron transfer
    rotates spin A about the transfer axis by angle phi with sin(phi)=P.
    rho(0) = R_A P_S R_A^dagger (electron part), nucleus maximally mixed."""
    phi = np.arcsin(max(min(P, 0.999), 0.0))
    RA = expm_herm(-1j*phi*SAy)
    Pe = PS / np.real(np.trace(PS))          # normalized singlet (electron+nuc)
    rho0 = RA @ Pe @ RA.conj().T
    rho0 = rho0/np.real(np.trace(rho0))
    L = liouvillian(B_mT)
    rho_int = -np.linalg.solve(L, rho0.reshape(-1)).reshape(8,8)
    phiS = kS*np.real(np.trace(PS@rho_int))
    phiT = kT*np.real(np.trace(PT@rho_int))
    return phiT/(phiS+phiT), phiS/(phiS+phiT)

if __name__ == "__main__":
    Bs = np.linspace(0, 10, 61)
    fig, ax = plt.subplots(figsize=(6.4,4.2))
    print(f"{'P':>5} {'PhiT(0)':>9} {'PhiT(1mT)':>10} {'PhiT(5mT)':>10}  (superoxide fraction)")
    for P, style in ((0.0,'-'), (0.35,'--'), (0.70,':')):
        yt = np.array([yields(b, P)[0] for b in Bs])
        ax.plot(Bs, 100*yt, style, lw=2,
                label=f"CISS P = {P:.2f}")
        g = lambda b: yields(b,P)[0]
        print(f"{P:>5.2f} {g(0.0):>9.4f} {g(1.0):>10.4f} {g(5.0):>10.4f}")
    ax.set_xlabel("static magnetic field  B  [mT]")
    ax.set_ylabel(r"superoxide (triplet) yield $\Phi_T$  [%]")
    ax.set_title("CISS-biased radical-pair superoxide vs magnetic field")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("/Users/oleg/Documents/NVG-Research/biophysics_ciss_ros/fig_ros_vs_B.pdf")
    fig.savefig("/Users/oleg/Documents/NVG-Research/biophysics_ciss_ros/fig_ros_vs_B.png", dpi=150)
    # low-field-effect amplitude
    p0 = yields(0.0,0.0)[0]; p1 = yields(1.0,0.0)[0]
    print(f"\nLow-field effect (P=0): PhiT changes {100*(p1-p0):+.2f}% abs from 0 to 1 mT")
    print("figure written: fig_ros_vs_B.pdf")
