"""
1D wave equation, state (u, v=u_t) stored and compressed TOGETHER.

Answers the question: when the field and its time-derivative are stored as one
compressed record, wave energy E = KE + PE splits into two quadratic invariants
that live on DIFFERENT parts of the state, and an affine correction hits each
independently -- no coupling matrix, no iteration.

    KE = 1/2 * sum_i v_i^2                    (a sum of squares of v)
    PE = 1/2 * c^2 * sum_i (u_{i+1}-u_i)^2    (a sum of squares of du/dx)

Consequences that make the affine form exact here:
  * KE depends only on v  -> a multiplicative scale a_v on v fixes KE.
  * PE depends only on differences of u -> a multiplicative scale a_u on u
    fixes PE (a difference scales linearly with a_u).
  * A constant offset b added to u leaves every difference unchanged, so it
    does NOT touch PE; it only moves mass sum_i u_i. So b conserves mass
    without disturbing energy.
Three scalars (a_u, a_v, b), fully decoupled, restore {PE, KE, mass} exactly
because each invariant sees only its own parameter. That decoupling is the
whole point of the "stored together" case.

We run it in-loop: compress the state every step, and compare drift with and
without the correction over a long integration.
"""
import numpy as np
np.random.seed(0)

# --- grid / physics ---------------------------------------------------------
N   = 2048
L   = 1.0
dx  = L / N
c   = 1.0
dt  = 0.5 * dx / c          # CFL = 0.5
NSTEPS = 20000
x   = np.arange(N) * dx

def energies(u, v):
    du = np.roll(u, -1) - u                # periodic forward difference
    KE = 0.5 * np.sum(v * v)
    PE = 0.5 * c * c * np.sum(du * du) / (dx * dx)
    return KE, PE, KE + PE

def mass(u):
    return np.sum(u)

# --- lossy step: fixed-step truncation quantization (biased, like a codec) --
def quantize(a, q):
    return np.floor(a / q) * q

# affine correction of the STORED STATE (u,v) to restore PE, KE, mass
def correct_state(u_q, v_q, PE0, KE0, M0):
    # KE depends only on v -> scale v so KE matches (KE ~ a_v^2)
    KEq, PEq, _ = energies(u_q, v_q)
    a_v = np.sqrt(KE0 / KEq) if KEq > 0 else 1.0
    v_c = a_v * v_q
    # PE depends only on differences of u -> scale u so PE matches (PE ~ a_u^2).
    # Offset-invariant, so apply the scale before the mass offset.
    a_u = np.sqrt(PE0 / PEq) if PEq > 0 else 1.0
    u_c = a_u * u_q
    # mass: a constant offset moves sum(u) but leaves every difference (PE) intact
    b = (M0 - np.sum(u_c)) / N
    u_c = u_c + b
    return u_c, v_c, (a_u, a_v, b)

def run(mode, q=2e-3):
    # initial condition: a RIGHT-TRAVELING gaussian pulse (nonzero v so KE is
    # live from the start) plus low-mode structure so mass and PE are nonzero.
    sig = 0.04
    g = np.exp(-((x - 0.5) ** 2) / (2 * sig ** 2))
    u = g + 0.05 * np.sin(2 * np.pi * x / L) + 0.03 * np.sin(6 * np.pi * x / L)
    v = c * (x - 0.5) / (sig ** 2) * g          # u_t for a right-moving pulse

    KE0, PE0, E0 = energies(u, v)
    M0 = mass(u)
    Efirst = E0

    drift_hist, step_hist = [], []
    blew = False
    for step in range(1, NSTEPS + 1):
        # leapfrog on the wave equation (periodic)
        lap = np.roll(u, -1) - 2 * u + np.roll(u, 1)
        v = v + dt * (c * c / (dx * dx)) * lap
        u = u + dt * v

        if mode != 'baseline':
            # true invariants of the current (pre-compression) state
            KEt, PEt, _ = energies(u, v)
            Mt = mass(u)
            u_q = quantize(u, q); v_q = quantize(v, q)
            if mode == 'lossy+correct':
                u, v, _ = correct_state(u_q, v_q, PEt, KEt, Mt)
            else:
                u, v = u_q, v_q

        if step % 100 == 0:
            _, _, E = energies(u, v)
            d = abs(E - Efirst) / Efirst
            drift_hist.append(d); step_hist.append(step)
            if not np.isfinite(E) or E > 50 * Efirst:
                blew = True
                break
    return dict(mode=mode, drift=drift_hist, step=step_hist,
                blew=blew, laststep=step, E0=Efirst)

if __name__ == "__main__":
    # ---- (1) THE ANSWER: single stored record (u,v) compressed together -----
    print("=== single compressed record (u,v) stored together, q=2e-3 ===")
    xg = np.arange(N) * dx
    sig = 0.04
    g = np.exp(-((xg - 0.5) ** 2) / (2 * sig ** 2))
    u = g + 0.05 * np.sin(2 * np.pi * xg / L)
    v = c * (xg - 0.5) / (sig ** 2) * g            # u_t of a right-moving pulse
    KE0, PE0, E0 = energies(u, v); M0 = mass(u)
    q = 2e-3
    u_q = quantize(u, q); v_q = quantize(v, q)
    KEu, PEu, Eu = energies(u_q, v_q); Mu = mass(u_q)
    u_c, v_c, params = correct_state(u_q, v_q, PE0, KE0, M0)
    KEc, PEc, Ec = energies(u_c, v_c); Mc = mass(u_c)
    print("  %-10s %14s %14s %14s %12s" % ("", "KE", "PE", "E_total", "mass"))
    print("  %-10s %14.6f %14.6f %14.6f %12.5f" % ("original",  KE0, PE0, E0, M0))
    print("  %-10s %14.6f %14.6f %14.6f %12.5f" % ("lossy",     KEu, PEu, Eu, Mu))
    print("  %-10s %14.6f %14.6f %14.6f %12.5f" % ("corrected", KEc, PEc, Ec, Mc))
    print("  rel err  KE : lossy %.2e  ->  corrected %.2e" % (abs(KEu-KE0)/KE0, abs(KEc-KE0)/KE0))
    print("  rel err  PE : lossy %.2e  ->  corrected %.2e" % (abs(PEu-PE0)/PE0, abs(PEc-PE0)/PE0))
    print("  rel err  E  : lossy %.2e  ->  corrected %.2e" % (abs(Eu-E0)/E0,   abs(Ec-E0)/E0))
    print("  rel err mass: lossy %.2e  ->  corrected %.2e" % (abs(Mu-M0)/abs(M0), abs(Mc-M0)/abs(M0)))
    print("  three scalars, each on its own invariant:  a_u(PE)=%.6f  a_v(KE)=%.6f  b(mass)=%.3e"
          % params)

    # ---- (2) HONEST in-loop contrast: global correction != stability --------
    print("\n=== in-loop, compress every step, N=%d, %d steps (q=2e-3) ===" % (N, NSTEPS))
    print("  %-14s %16s %16s" % ("mode", "result", "E-drift @ last"))
    for mode in ('baseline', 'lossy', 'lossy+correct'):
        r = run(mode)
        ls = r['laststep']
        status = ('BLEW UP @%d' % ls) if r['blew'] else ('stable (%d)' % ls)
        fd = r['drift'][-1] if r['drift'] else float('nan')
        print("  %-14s %16s %16.3e" % (mode, status, fd))
    print("  (energy stays pinned by the correction, but undamped quantization")
    print("   noise in a non-dissipative scheme still grows -- a global scalar")
    print("   constraint does not control the per-mode error. Contrast the LBM,")
    print("   where per-NODE moment restore + dissipative collision stays stable.)")
