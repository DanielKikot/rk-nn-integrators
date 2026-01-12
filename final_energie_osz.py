import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from typing import Callable, Tuple

# -----------------------------------------------------------------------------
# 0) 64-bit für numerische Stabilität
# -----------------------------------------------------------------------------
jax.config.update("jax_enable_x64", True)

# -----------------------------------------------------------------------------
# 1) Isotroper harmonischer Oszillator: y=(q,p), y'=(p,-q)
#    Hamiltonian: H(q,p) = 1/2 (q^2 + p^2)
#    Exakter Fluss: Rotation um Winkel h
# -----------------------------------------------------------------------------
@jax.jit
def oscillator_f(y: jnp.ndarray) -> jnp.ndarray:
    q, p = y
    return jnp.array([p, -q])

@jax.jit
def oscillator_H(y: jnp.ndarray) -> jnp.ndarray:
    q, p = y
    return 0.5 * (q**2 + p**2)

@jax.jit
def exact_flow(y: jnp.ndarray, h: float) -> jnp.ndarray:
    c, s = jnp.cos(h), jnp.sin(h)
    R = jnp.array([[c, s],
                   [-s, c]])
    return R @ y

# -----------------------------------------------------------------------------
# 2) Referenz im Nenner: Heun-Verfahren (RK2, Ordnung 2)
#    (klassisches RK-Verfahren für \hat y_n^{(RK)})
# -----------------------------------------------------------------------------
@partial(jax.jit, static_argnames=["f"])
def rk_ref_heun_step(f: Callable, y: jnp.ndarray, h: float) -> jnp.ndarray:
    k1 = f(y)
    k2 = f(y + h * k1)
    return y + 0.5 * h * (k1 + k2)

# -----------------------------------------------------------------------------
# 3) Allgemeiner s-stufiger RK-NN-Integrator (explizit, strikt lower)
#    theta_a: (s, s-1)    (nur Einträge j < i werden benutzt)
#    theta_c: (s,)
# -----------------------------------------------------------------------------
@partial(jit, static_argnames=["f", "s"])
def rk_nn_integrator(
    f: Callable[[jnp.ndarray], jnp.ndarray],
    y0: jnp.ndarray,
    h: float,
    theta_a: jnp.ndarray,   # shape (s, s-1)
    theta_c: jnp.ndarray,   # shape (s,)
    s: int
) -> jnp.ndarray:
    ks = []
    for i in range(s):
        if i == 0:
            y_stage = y0
        else:
            contrib = jnp.sum(jnp.array([theta_a[i, j] * ks[j] for j in range(i)]), axis=0)
            y_stage = y0 + h * contrib
        ks.append(f(y_stage))
    ks = jnp.stack(ks)  # (s, d)
    return y0 + h * jnp.sum(theta_c[:, None] * ks, axis=0)

# -----------------------------------------------------------------------------
# 4) Loss-Komponenten: L_rel + lambda_energy * L_energy
#     - L_rel : skaliert relativ zu Referenz-RK (Heun), über N_steps Schritte gemittelt
#     - L_energy : relativer Energiefehler (bezogen auf H(y0)), über N_steps Schritte gemittelt
#
#     "Exakt" ist hier wirklich exakt: y_true wird über exact_flow propagiert.
# -----------------------------------------------------------------------------
def make_scalar_loss_components_rel_energy(
    f: Callable[[jnp.ndarray], jnp.ndarray],
    H: Callable[[jnp.ndarray], jnp.ndarray],
    s: int,
    N_steps: int = 10,
    delta_den: float = 1e-12,
    delta_energy: float = 1e-12
):
    @jax.jit
    def comps(
        y0: jnp.ndarray,
        h: float,
        theta_a: jnp.ndarray,
        theta_c: jnp.ndarray,
        key_ignored: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:

        y_true = y0
        y_nn   = y0
        y_ref  = y0

        H0 = H(y0)

        Lrel_sum = 0.0
        Lene_sum = 0.0

        for _ in range(N_steps):
            # Exakte Referenz
            y_true = exact_flow(y_true, h)

            # RK-NN
            y_nn = rk_nn_integrator(f, y_nn, h, theta_a, theta_c, s)

            # Referenz-RK im Nenner: Heun
            y_ref = rk_ref_heun_step(f, y_ref, h)

            # skaliertes L_rel
            num2 = jnp.sum((y_nn  - y_true) ** 2)
            den2 = jnp.sum((y_ref - y_true) ** 2) + delta_den
            Lrel_sum = Lrel_sum + (num2 / den2)

            # Energie-Term (relativ zur Anfangsenergie)
            E_rel = (H(y_nn) - H0) / (jnp.abs(H0) + delta_energy)
            Lene_sum = Lene_sum + (E_rel ** 2)

        L_rel = Lrel_sum / N_steps
        L_energy = Lene_sum / N_steps
        return L_rel, L_energy

    return comps

# -----------------------------------------------------------------------------
# 5) pack/unpack θ
# -----------------------------------------------------------------------------
def pack_thetas(a: jnp.ndarray, c: jnp.ndarray) -> np.ndarray:
    return np.concatenate([np.array(a).ravel(), np.array(c)])

def unpack_thetas(x: np.ndarray, s: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    d = s * (s - 1)  # theta_a has shape (s, s-1)
    a = x[:d].reshape((s, s - 1))
    c = x[d:]
    return jnp.array(a), jnp.array(c)

# -----------------------------------------------------------------------------
# 6) Batch-Loss & Trace (2 Komponenten)
# -----------------------------------------------------------------------------
def make_batch_loss_and_trace_2(scalar_components_fn: Callable):
    @jax.jit
    def batch_loss(y0s, hs, theta_a, theta_c, keys, lambda_energy):
        Lrel, Lene = vmap(lambda y, h, k: scalar_components_fn(y, h, theta_a, theta_c, k))(y0s, hs, keys)
        return jnp.mean(Lrel + lambda_energy * Lene)

    @jax.jit
    def batch_components(y0s, hs, theta_a, theta_c, keys):
        return vmap(lambda y, h, k: scalar_components_fn(y, h, theta_a, theta_c, k))(y0s, hs, keys)

    return batch_loss, batch_components

# -----------------------------------------------------------------------------
# 7) Training via (L-)BFGS
# -----------------------------------------------------------------------------
def train_bfgs_with_trace(
    y0s, hs,
    f, H,
    s: int,
    N_steps: int = 10,
    tol: float = 1e-6,
    maxiter: int = 1000,
    lambda_energy: float = 1.0,
    method: str = "L-BFGS-B"
):
    B = y0s.shape[0]
    keys = jax.random.split(jax.random.PRNGKey(0), B)  # unbenutzt, nur Signatur-kompatibel

    scalar_comps = make_scalar_loss_components_rel_energy(
        f=f, H=H, s=s, N_steps=N_steps, delta_den=1e-12, delta_energy=1e-12
    )
    batch_loss, batch_comps = make_batch_loss_and_trace_2(scalar_comps)

    hist_rel, hist_energy = [], []

    def callback(xk):
        a, c = unpack_thetas(xk, s)
        Lr, Le = batch_comps(y0s, hs, a, c, keys)
        hist_rel.append(float(jnp.mean(Lr)))
        hist_energy.append(float(jnp.mean(Le)))

    # Initialisierung
    ka, kc = jax.random.split(jax.random.PRNGKey(1))
    a0 = jax.random.normal(ka, (s, s - 1)) * 0.1
    c0 = jax.random.normal(kc, (s,)) * 0.1
    x0 = pack_thetas(a0, c0)

    # Anfangswerte in Historie
    Lr0, Le0 = batch_comps(y0s, hs, a0, c0, keys)
    hist_rel.append(float(jnp.mean(Lr0)))
    hist_energy.append(float(jnp.mean(Le0)))

    def obj(x_flat):
        a, c = unpack_thetas(x_flat, s)
        val = batch_loss(y0s, hs, a, c, keys, lambda_energy)
        vf = float(val)
        if not np.isfinite(vf):
            raise ValueError("Loss became non-finite.")
        return vf

    def grad(x_flat):
        g = jax.grad(lambda xf: batch_loss(y0s, hs, *unpack_thetas(xf, s), keys, lambda_energy))(jnp.array(x_flat))
        return np.array(g)

    res = optimize.minimize(
        fun=obj,
        x0=x0,
        jac=grad,
        method=method,
        callback=callback,
        options={'gtol': tol, 'maxiter': maxiter}
    )
    print("Converged:", res.success, "status:", res.status, "iters:", res.nit, res.message)

    a_star, c_star = unpack_thetas(res.x, s)
    return a_star, c_star, (hist_rel, hist_energy), res

# -----------------------------------------------------------------------------
# 8) Wrapper + Beispielaufruf: Harm. Oszillator
#    K=500, q0,p0 in [-2,2], h in [0.1,0.2], s=4, N_steps=10
# -----------------------------------------------------------------------------
def algorithm_oscillator_rel_energy(
    y0s, hs,
    s: int = 4,
    N_steps: int = 10,
    tol: float = 1e-6,
    maxiter: int = 1000,
    lambda_energy: float = 1.0,
    method: str = "L-BFGS-B"
):
    f = oscillator_f
    H = oscillator_H
    return train_bfgs_with_trace(
        y0s, hs,
        f=f, H=H,
        s=s, N_steps=N_steps,
        tol=tol, maxiter=maxiter,
        lambda_energy=lambda_energy,
        method=method
    )

if __name__ == "__main__":
    # Datensatz: q0,p0 in [-2,2], h in [0.1,0.2]
    K = 500
    key = jax.random.PRNGKey(0)
    key, k1 = jax.random.split(key)
    q0 = jax.random.uniform(k1, (K,), minval=-2.0, maxval=2.0)
    key, k2 = jax.random.split(key)
    p0 = jax.random.uniform(k2, (K,), minval=-2.0, maxval=2.0)
    y0s = jnp.stack([q0, p0], axis=1)

    key, k3 = jax.random.split(key)
    hs = jax.random.uniform(k3, (K,), minval=0.1, maxval=0.2)

    # Training
    a_star, c_star, (hist_rel, hist_energy), res = algorithm_oscillator_rel_energy(
        y0s, hs,
        s=4,
        N_steps=10,
        tol=1e-6,
        maxiter=1000,
        lambda_energy=50,     # ggf. erhöhen, falls Energie stärker priorisiert werden soll
        method="L-BFGS-B"
    )

    print("theta_a:\n", a_star)
    print("theta_c:\n", c_star)

    # Verlauf plotten
    its = np.arange(len(hist_rel))
    plt.figure()
    plt.plot(its, hist_rel, marker='o', label=r'$L_{\mathrm{rel}}$')
    plt.plot(its, hist_energy, marker='o', label=r'$L_{\mathrm{energie}}$')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()
