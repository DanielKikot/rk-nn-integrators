import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from scipy import optimize
import matplotlib.pyplot as plt
from typing import Callable, Tuple

jax.config.update("jax_enable_x64", True)


# 1) Harmonischer Oszillator: f, exakter Fluss, Involution rho

@jax.jit
def oscillator_f(y: jnp.ndarray) -> jnp.ndarray:
    q, p = y
    return jnp.array([p, -q])

@jax.jit
def exact_flow(y: jnp.ndarray, h: float) -> jnp.ndarray:
    c, s = jnp.cos(h), jnp.sin(h)
    R = jnp.array([[c, s],
                   [-s, c]])
    return R @ y

@jax.jit
def rho(y: jnp.ndarray) -> jnp.ndarray:
    q, p = y
    return jnp.array([q, -p])


# 2) Referenz im Nenner: explizites Euler

@partial(jax.jit, static_argnames=["f"])
def rk_ref_euler_step(f: Callable, y: jnp.ndarray, h: float) -> jnp.ndarray:
    return y + h * f(y)


# 3) Allgemeiner s-stufiger RK-NN (explizit)

@partial(jit, static_argnames=["f", "s"])
def rk_nn_integrator(
    f: Callable[[jnp.ndarray], jnp.ndarray],
    y0: jnp.ndarray,
    h: float,
    theta_a: jnp.ndarray,
    theta_c: jnp.ndarray,
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
    ks = jnp.stack(ks)
    return y0 + h * jnp.sum(theta_c[:, None] * ks, axis=0)


# 4) Scalar Loss pro Sample: (L_rel, L_rev, num_mean, den_mean)

def make_scalar_loss_components_rel_rev(
    f: Callable[[jnp.ndarray], jnp.ndarray],
    s: int,
    N_steps: int = 3,
    delta_den: float = 1e-12,
):
    @jax.jit
    def comps(
        y0: jnp.ndarray,
        h: float,
        theta_a: jnp.ndarray,
        theta_c: jnp.ndarray,
        key_ignored: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

        y_true = y0
        y_nn   = y0
        y_ref  = y0

        # für L_rev brauchen wir y_{n-1}
        y_prev = y0

        Lrel_sum = 0.0
        Lrev_sum = 0.0
        num_sum  = 0.0
        den_sum  = 0.0

        for _ in range(N_steps):
            # exakte Referenz
            y_true = exact_flow(y_true, h)

            # RK-NN
            y_nn = rk_nn_integrator(f, y_nn, h, theta_a, theta_c, s)

            # Referenz-RK im Nenner: Euler
            y_ref = rk_ref_euler_step(f, y_ref, h)

            # L_rel: num/den
            num2 = jnp.sum((y_nn  - y_true) ** 2)
            den2 = jnp.sum((y_ref - y_true) ** 2) + delta_den

            num_sum = num_sum + num2
            den_sum = den_sum + den2
            Lrel_sum = Lrel_sum + (num2 / den2)

            # L_rev: Phi_h(rho(y_n)) ≈ rho(y_{n-1})
            y_back = rk_nn_integrator(f, rho(y_nn), h, theta_a, theta_c, s)
            res_rev = y_back - rho(y_prev)
            Lrev_sum = Lrev_sum + jnp.sum(res_rev ** 2)

            # update y_{n-1}
            y_prev = y_nn

        L_rel = Lrel_sum / N_steps
        L_rev = Lrev_sum / N_steps
        num_mean = num_sum / N_steps
        den_mean = den_sum / N_steps
        return L_rel, L_rev, num_mean, den_mean

    return comps


# 5) pack/unpack θ
def pack_thetas(a: jnp.ndarray, c: jnp.ndarray) -> np.ndarray:
    return np.concatenate([np.array(a).ravel(), np.array(c)])

def unpack_thetas(x: np.ndarray, s: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    dA = s * (s - 1)
    a = x[:dA].reshape((s, s - 1))
    c = x[dA:]
    return jnp.array(a), jnp.array(c)

# 6) Batch-Loss & Trace
def make_batch_loss_and_trace_rel_rev(scalar_components_fn: Callable):
    @jax.jit
    def batch_components(y0s, hs, theta_a, theta_c, keys):

        return vmap(lambda y, h, k: scalar_components_fn(y, h, theta_a, theta_c, k))(y0s, hs, keys)

    @jax.jit
    def batch_loss(y0s, hs, theta_a, theta_c, keys, lambda_rev):
        Lrel, Lrev, num_mean, den_mean = batch_components(y0s, hs, theta_a, theta_c, keys)
        return jnp.mean(Lrel + lambda_rev * Lrev)

    return batch_loss, batch_components


# 7) Training via SciPy BFGS
def train_bfgs_with_trace(
    y0s: jnp.ndarray,
    hs: jnp.ndarray,
    s: int = 4,
    N_steps: int = 3,
    lambda_rev: float = 1.0,
    tol: float = 1e-6,
    maxiter: int = 500,
    method: str = "L-BFGS-B",
    print_every: int = 1,
):
    K = y0s.shape[0]
    keys = jax.random.split(jax.random.PRNGKey(0), K)

    scalar_comps = make_scalar_loss_components_rel_rev(
        f=oscillator_f, s=s, N_steps=N_steps, delta_den=1e-12
    )
    batch_loss, batch_comps = make_batch_loss_and_trace_rel_rev(scalar_comps)

    hist_rel, hist_rev, hist_num, hist_den = [], [], [], []
    it_counter = {"it": 0}

    def callback(xk):
        a, c = unpack_thetas(xk, s)
        Lrel, Lrev, num_mean, den_mean = batch_comps(y0s, hs, a, c, keys)

        mean_Lrel = float(jnp.mean(Lrel))
        mean_Lrev = float(jnp.mean(Lrev))
        mean_num  = float(jnp.mean(num_mean))
        mean_den  = float(jnp.mean(den_mean))

        hist_rel.append(mean_Lrel)
        hist_rev.append(mean_Lrev)
        hist_num.append(mean_num)
        hist_den.append(mean_den)

        if it_counter["it"] % print_every == 0:
            print(
                f"iter={it_counter['it']:4d} | "
                f"mean L_rel={mean_Lrel:.3e} | mean num={mean_num:.3e} | mean den={mean_den:.3e} | "
                f"mean L_rev={mean_Lrev:.3e}"
            )
        it_counter["it"] += 1


    ka, kc = jax.random.split(jax.random.PRNGKey(1))
    a0 = jax.random.normal(ka, (s, s - 1)) * 0.1
    c0 = jax.random.normal(kc, (s,)) * 0.1
    x0 = pack_thetas(a0, c0)


    Lrel0, Lrev0, num0, den0 = batch_comps(y0s, hs, a0, c0, keys)
    print(
        f"init | mean L_rel={float(jnp.mean(Lrel0)):.3e} | mean num={float(jnp.mean(num0)):.3e} "
        f"| mean den={float(jnp.mean(den0)):.3e} | mean L_rev={float(jnp.mean(Lrev0)):.3e}"
    )

    def obj(x_flat):
        a, c = unpack_thetas(x_flat, s)
        val = batch_loss(y0s, hs, a, c, keys, lambda_rev)
        vf = float(val)
        if not np.isfinite(vf):
            raise ValueError("Loss became non-finite.")
        return vf

    def grad(x_flat):
        g = jax.grad(lambda xf: batch_loss(y0s, hs, *unpack_thetas(xf, s), keys, lambda_rev))(jnp.array(x_flat))
        return np.array(g)

    res = optimize.minimize(
        fun=obj,
        x0=x0,
        jac=grad,
        method=method,
        callback=callback,
        options={"gtol": tol, "maxiter": maxiter},
    )
    print("Converged:", res.success, "status:", res.status, "iters:", res.nit, res.message)

    a_star, c_star = unpack_thetas(res.x, s)
    return a_star, c_star, (hist_rel, hist_rev, hist_num, hist_den), res


# 8) Datensatz
def make_dataset(K: int = 500, h_min: float = 0.1, h_max: float = 0.2, seed: int = 0):
    key = jax.random.PRNGKey(seed)
    key, k1 = jax.random.split(key)
    q0 = jax.random.uniform(k1, (K,), minval=-2.0, maxval=2.0)
    key, k2 = jax.random.split(key)
    p0 = jax.random.uniform(k2, (K,), minval=-2.0, maxval=2.0)
    y0s = jnp.stack([q0, p0], axis=1)

    key, k3 = jax.random.split(key)
    hs = jax.random.uniform(k3, (K,), minval=h_min, maxval=h_max)
    return y0s, hs

if __name__ == "__main__":
    K = 500
    s = 4
    N_steps = 3

    y0s, hs = make_dataset(K=K, h_min=0.1, h_max=0.2, seed=0)

    a_star, c_star, (hist_rel, hist_rev, hist_num, hist_den), res = train_bfgs_with_trace(
        y0s=y0s,
        hs=hs,
        s=s,
        N_steps=N_steps,
        lambda_rev=1.0,   # ggf. erhöhen, falls Reversibilität stärker priorisiert werden soll
        tol=1e-6,
        maxiter=500,
        method="L-BFGS-B",
        print_every=1
    )

    print("theta_a:\n", np.array(a_star))
    print("theta_c:\n", np.array(c_star))

    # Verlauf plotten
    its = np.arange(len(hist_rel))

    plt.figure()
    plt.plot(its, hist_rel, marker="o", label=r"$\overline{L}_{\mathrm{rel}}$")
    plt.plot(its, hist_rev, marker="o", label=r"$\overline{L}_{\mathrm{rev}}$")
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(its, hist_num, marker="o", label="mean num = E||y_nn - y_true||^2")
    plt.plot(its, hist_den, marker="o", label="mean den = E||y_euler - y_true||^2")
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Wert")
    plt.legend()
    plt.tight_layout()
    plt.show()
