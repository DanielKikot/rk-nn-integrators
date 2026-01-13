import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from typing import Callable, Tuple



# 0) 64-bit für numerische Stabilität
jax.config.update("jax_enable_x64", True)

# 1) Mathematisches Pendel: y=(q,p)
#    Hamiltonian: H(q,p)= p^2/(2 m l^2) + m g l (1 - cos q)
#    Hamilton-Gleichungen: qdot = p/(m l^2), pdot = -m g l sin q
# -----------------------------------------------------------------------------
@jax.jit
def pendulum_f(y: jnp.ndarray, g: float = 9.81, m: float = 1.0, ell: float = 1.0) -> jnp.ndarray:
    q, p = y
    return jnp.array([p / (m * ell**2), -m * g * ell * jnp.sin(q)])

@jax.jit
def pendulum_H(y: jnp.ndarray, g: float = 9.81, m: float = 1.0, ell: float = 1.0) -> jnp.ndarray:
    q, p = y
    return (p**2) / (2.0 * m * ell**2) + m * g * ell * (1.0 - jnp.cos(q))


# 2) Referenz im Nenner: explizites Euler-Verfahren (klassisches RK-Verfahren), für approximation von y_true: RK4
@partial(jax.jit, static_argnames=["f"])
def rk_ref_euler_step(f: Callable, y: jnp.ndarray, h: float) -> jnp.ndarray:
    return y + h * f(y)

@partial(jax.jit, static_argnames=["f"])
def rk4_step(f: Callable, y: jnp.ndarray, h: float) -> jnp.ndarray:
    k1 = f(y)
    k2 = f(y + 0.5 * h * k1)
    k3 = f(y + 0.5 * h * k2)
    k4 = f(y + h * k3)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# 3) Allgemeiner s-stufiger RK-NN-Integrator (explizit)
#    theta_a: (s, s-1)    (nur Einträge j < i werden benutzt)
#    theta_c: (s,)
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
            # contrib = sum_{j=0..i-1} theta_a[i,j] * k_j
            contrib = jnp.sum(jnp.array([theta_a[i, j] * ks[j] for j in range(i)]), axis=0)
            y_stage = y0 + h * contrib
        ks.append(f(y_stage))
    ks = jnp.stack(ks)
    return y0 + h * jnp.sum(theta_c[:, None] * ks, axis=0)


# 4) Loss-Komponenten: L_rel + lambda_energy * L_energy
#     - L_rel : skaliert relativ zu Referenz-RK (Euler), über N_steps Schritte gemittelt
#     - L_energy : relativer Energiefehler (bezogen auf H(y0)), über N_steps Schritte gemittelt

def make_scalar_loss_components_rel_energy(
    f: Callable[[jnp.ndarray], jnp.ndarray],
    H: Callable[[jnp.ndarray], jnp.ndarray],
    s: int,
    N_steps: int = 5,
    delta_den: float = 1e-12,       # Stabilisierung des Nenners in L_rel
    delta_energy: float = 1e-12     # Stabilisierung falls H(y0) ~ 0
):
    """
    Gibt (L_rel, L_energy) zurück.
    """
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
            # "Exakte" Referenz (RK4)
            y_true = rk4_step(f, y_true, h)

            # RK-NN
            y_nn = rk_nn_integrator(f, y_nn, h, theta_a, theta_c, s)

            # Referenz-RK im Nenner (Euler)
            y_ref = rk_ref_euler_step(f, y_ref, h)

            # L_rel
            num2 = jnp.sum((y_nn  - y_true) ** 2)
            den2 = jnp.sum((y_ref - y_true) ** 2) + delta_den
            Lrel_sum = Lrel_sum + (num2 / den2)

            # Energie-Term
            E_rel = (H(y_nn) - H0) / (jnp.abs(H0) + delta_energy)
            Lene_sum = Lene_sum + (E_rel ** 2)

        L_rel = Lrel_sum / N_steps
        L_energy = Lene_sum / N_steps
        return L_rel, L_energy

    return comps


# 5) pack/unpack θ
def pack_thetas(a: jnp.ndarray, c: jnp.ndarray) -> np.ndarray:
    return np.concatenate([np.array(a).ravel(), np.array(c)])

def unpack_thetas(x: np.ndarray, s: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    d = s * (s - 1)  # because theta_a has shape (s, s-1)
    a = x[:d].reshape((s, s - 1))
    c = x[d:]
    return jnp.array(a), jnp.array(c)


# 6) Batch-Loss & Trace (2 Komponenten)
def make_batch_loss_and_trace_2(scalar_components_fn: Callable):
    @jax.jit
    def batch_loss(y0s, hs, theta_a, theta_c, keys, lambda_energy):
        Lrel, Lene = vmap(lambda y, h, k: scalar_components_fn(y, h, theta_a, theta_c, k))(y0s, hs, keys)
        return jnp.mean(Lrel + lambda_energy * Lene)

    @jax.jit
    def batch_components(y0s, hs, theta_a, theta_c, keys):
        return vmap(lambda y, h, k: scalar_components_fn(y, h, theta_a, theta_c, k))(y0s, hs, keys)

    return batch_loss, batch_components


# 7) Training mit (L_rel, L_energy) und Verlauf via BFGS
def train_bfgs_with_trace(
    y0s, hs,
    f, H,
    s: int,
    N_steps: int = 5,
    tol: float = 1e-6,
    maxiter: int = 500,
    lambda_energy: float = 1.0,
    method: str = "BFGS"
):
    B = y0s.shape[0]
    keys = jax.random.split(jax.random.PRNGKey(0), B)  # unbenutzt, aber Signatur-kompatibel

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
        # Gradienten über JAX, zurück als numpy für SciPy
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


# 8) Wrapper & Beispielaufruf
def algorithm_pendulum_rel_energy(
    y0s, hs,
    s: int = 4,
    N_steps: int = 5,
    tol: float = 1e-6,
    maxiter: int = 1000,
    lambda_energy: float = 100.0,
    g: float = 9.81,
    m: float = 1.0,
    ell: float = 1.0,
    method: str = "L-BFGS-B"
):
    f = lambda y: pendulum_f(y, g=g, m=m, ell=ell)
    H = lambda y: pendulum_H(y, g=g, m=m, ell=ell)
    return train_bfgs_with_trace(
        y0s, hs,
        f=f, H=H,
        s=s, N_steps=N_steps,
        tol=tol, maxiter=maxiter,
        lambda_energy=lambda_energy,
        method=method
    )

if __name__ == "__main__":
    # Datensatz:
    K = 100
    key = jax.random.PRNGKey(0)
    key, k1 = jax.random.split(key)
    q0 = jax.random.uniform(k1, (K,), minval=-jnp.pi, maxval=jnp.pi)
    key, k2 = jax.random.split(key)
    p0 = jax.random.uniform(k2, (K,), minval=-2.0, maxval=2.0)
    y0s = jnp.stack([q0, p0], axis=1)

    key, k3 = jax.random.split(key)
    hs = jax.random.uniform(k3, (K,), minval=0.01, maxval=0.2)

    # Training: skaliertes L_rel (Nenner Euler) + lambda_energy * Energie-Term
    a_star, c_star, (hist_rel, hist_energy), res = algorithm_pendulum_rel_energy(
        y0s, hs,
        s=4,            # RK-NN Stufen
        N_steps=5,      # Anzahl Zeitschritte
        tol=1e-6,
        maxiter=1000,
        lambda_energy=1.0,
        g=9.81, m=1.0, ell=1.0,
        method="L-BFGS-B"
    )

    print("theta_a:\n", a_star)
    print("theta_c:\n", c_star)

    # Verlauf plotten
    its = np.arange(len(hist_rel))
    plt.figure()
    plt.plot(its, hist_rel, marker='o', label='L')
    plt.plot(its, hist_energy, marker='o', label='L_energie')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Gesamtfehler über den Datensatz')
    plt.legend()
    plt.tight_layout()
    plt.show()
