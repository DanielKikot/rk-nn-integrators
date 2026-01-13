import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from scipy import optimize
import matplotlib.pyplot as plt
from typing import Callable, Tuple

jax.config.update("jax_enable_x64", True)


# Problemparameter
d = 2
m1 = 1.0
m2 = 1.0
G = 1.0

mu = (m1 * m2) / (m1 + m2)   # 0.5
kappa = G * m1 * m2          # 1.0

soft_eps = 1e-12

# 1) Vektorfeld & Hamiltonian
@jax.jit
def two_body_f(y: jnp.ndarray) -> jnp.ndarray:
    q = y[:d]
    p = y[d:]
    r2 = jnp.dot(q, q) + soft_eps
    r = jnp.sqrt(r2)
    qdot = p / mu
    pdot = -kappa * q / (r**3)
    return jnp.concatenate([qdot, pdot])

@jax.jit
def two_body_H(y: jnp.ndarray) -> jnp.ndarray:
    q = y[:d]
    p = y[d:]
    r = jnp.sqrt(jnp.dot(q, q) + soft_eps)
    T = 0.5 * jnp.dot(p, p) / mu
    V = -kappa / r
    return T + V

# 2) Heun (RK2) Referenz im Nenner
@partial(jax.jit, static_argnames=["f"])
def heun_step(f: Callable, y: jnp.ndarray, h: float) -> jnp.ndarray:
    k1 = f(y)
    k2 = f(y + h * k1)
    return y + 0.5 * h * (k1 + k2)

# 3) RK4 und RK4-Substep Referenz (y_true)
@partial(jax.jit, static_argnames=["f"])
def rk4_step(f: Callable, y: jnp.ndarray, h: float) -> jnp.ndarray:
    k1 = f(y)
    k2 = f(y + 0.5 * h * k1)
    k3 = f(y + 0.5 * h * k2)
    k4 = f(y + h * k3)
    return y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

@partial(jax.jit, static_argnames=["f", "n_ref"])
def rk4_ref_step(f: Callable, y: jnp.ndarray, h: float, n_ref: int = 100) -> jnp.ndarray:
    dt = h / n_ref
    def body(i, state):
        return rk4_step(f, state, dt)
    return jax.lax.fori_loop(0, n_ref, body, y)


# 4) Rotation in 2D: auf q und p anwenden
@jax.jit
def rot2_matrix(theta: jnp.ndarray) -> jnp.ndarray:
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([[c, -s],
                      [s,  c]])

@jax.jit
def apply_rotation(y: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    R = rot2_matrix(theta)
    q = y[:d]
    p = y[d:]
    return jnp.concatenate([R @ q, R @ p])


# 5) RK-NN Schritt (explizit, s-stufig)
# theta_a: (s, s-1), theta_c: (s,)
@partial(jit, static_argnames=["f", "s_stages"])
def rk_nn_step(
    f: Callable[[jnp.ndarray], jnp.ndarray],
    y0: jnp.ndarray,
    h: float,
    theta_a: jnp.ndarray,
    theta_c: jnp.ndarray,
    s_stages: int
) -> jnp.ndarray:
    ks = []
    for i in range(s_stages):
        if i == 0:
            y_stage = y0
        else:
            contrib = jnp.sum(jnp.array([theta_a[i, j] * ks[j] for j in range(i)]), axis=0)
            y_stage = y0 + h * contrib
        ks.append(f(y_stage))
    ks = jnp.stack(ks)
    return y0 + h * jnp.sum(theta_c[:, None] * ks, axis=0)


# 6) Loss pro Sample: (L_rel, L_rot, num_mean, den_mean)
def make_scalar_loss_components_rel_rot(
    f: Callable[[jnp.ndarray], jnp.ndarray],
    s_stages: int,
    N_steps: int = 1,
    J_rot: int = 5,
    n_ref: int = 100,
    delta_den: float = 1e-12,
):
    @jax.jit
    def comps(
        y0: jnp.ndarray,
        h: float,
        theta_a: jnp.ndarray,
        theta_c: jnp.ndarray,
        angles: jnp.ndarray  # (N_steps, J_rot)
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

        y_true = y0
        y_nn   = y0
        y_ref  = y0

        Lrel_sum = 0.0
        Lrot_sum = 0.0
        num_sum  = 0.0
        den_sum  = 0.0

        for n in range(N_steps):

            y_true = rk4_ref_step(f, y_true, h, n_ref=n_ref)
            y_nn   = rk_nn_step(f, y_nn, h, theta_a, theta_c, s_stages)
            y_ref  = heun_step(f, y_ref, h)

            # num/den for L_rel
            num2 = jnp.sum((y_nn - y_true) ** 2)
            den2 = jnp.sum((y_ref - y_true) ** 2) + delta_den

            num_sum = num_sum + num2
            den_sum = den_sum + den2
            Lrel_sum = Lrel_sum + (num2 / den2)


            y_nn_next = rk_nn_step(f, y_nn, h, theta_a, theta_c, s_stages)

            def one_rot_loss(theta_angle):
                lhs = rk_nn_step(f, apply_rotation(y_nn, theta_angle), h, theta_a, theta_c, s_stages)
                rhs = apply_rotation(y_nn_next, theta_angle)
                return jnp.sum((lhs - rhs) ** 2)

            Lrot_n = jnp.mean(jax.vmap(one_rot_loss)(angles[n]))
            Lrot_sum = Lrot_sum + Lrot_n

        Lrel = Lrel_sum / N_steps
        Lrot = Lrot_sum / N_steps
        num_mean = num_sum / N_steps
        den_mean = den_sum / N_steps
        return Lrel, Lrot, num_mean, den_mean

    return comps

# 7) pack/unpack θ
def pack_thetas(a: jnp.ndarray, c: jnp.ndarray) -> np.ndarray:
    return np.concatenate([np.array(a).ravel(), np.array(c)])

def unpack_thetas(x: np.ndarray, s_stages: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    dA = s_stages * (s_stages - 1)  # theta_a shape (s, s-1)
    a = x[:dA].reshape((s_stages, s_stages - 1))
    c = x[dA:]
    return jnp.array(a), jnp.array(c)

# 8) Batch loss + components
def make_batch_loss(s_stages: int, N_steps: int, J_rot: int, n_ref: int):
    scalar = make_scalar_loss_components_rel_rot(
        f=two_body_f, s_stages=s_stages, N_steps=N_steps, J_rot=J_rot, n_ref=n_ref
    )

    @jax.jit
    def batch_components(y0s, hs, theta_a, theta_c, angles_all):

        return vmap(lambda y0, h, ang: scalar(y0, h, theta_a, theta_c, ang))(y0s, hs, angles_all)

    @jax.jit
    def batch_loss(y0s, hs, theta_a, theta_c, angles_all, lambda_rot):
        Lrel, Lrot, num_mean, den_mean = batch_components(y0s, hs, theta_a, theta_c, angles_all)
        return jnp.mean(Lrel + lambda_rot * Lrot)

    return batch_loss, batch_components


# 9) Dataset
def make_dataset(K: int = 500, h_min: float = 0.05, h_max: float = 0.2, r_min: float = 1.5, seed: int = 0):
    rng = np.random.default_rng(seed)

    q0 = rng.uniform(-2.0, 2.0, size=(K, d))
    p0 = rng.uniform(-2.0, 2.0, size=(K, d))


    norms = np.linalg.norm(q0, axis=1)
    scale = np.maximum(1.0, r_min / (norms + 1e-12))
    q0 = q0 * scale[:, None]

    y0 = np.concatenate([q0, p0], axis=1)  # (K, 2d)
    h = rng.uniform(h_min, h_max, size=(K,))
    return jnp.array(y0), jnp.array(h)


# 10) Zufällige Rotationen pro Iteration

def sample_angles(rng: np.random.Generator, K: int, N_steps: int, J_rot: int):
    return jnp.array(rng.uniform(0.0, 2*np.pi, size=(K, N_steps, J_rot)))


# 11) Training BFGS
def train_rknn_two_body(
    y0s: jnp.ndarray,
    hs: jnp.ndarray,
    s_stages: int = 3,
    N_steps: int = 1,
    J_rot: int = 5,
    n_ref: int = 100,
    lambda_rot: float = 1.0,
    tol: float = 1e-6,
    maxiter: int = 500,
    method: str = "L-BFGS-B",
    angles_seed: int = 42,
    print_every: int = 1,
):
    K = y0s.shape[0]
    batch_loss, batch_components = make_batch_loss(s_stages, N_steps, J_rot, n_ref)

    rng_angles = np.random.default_rng(angles_seed)
    angles_holder = {"angles": sample_angles(rng_angles, K, N_steps, J_rot)}
    it_counter = {"it": 0}

    hist_rel, hist_rot, hist_num, hist_den = [], [], [], []

    def callback(xk):

        a, c = unpack_thetas(xk, s_stages)
        Lrel, Lrot, num_mean, den_mean = batch_components(y0s, hs, a, c, angles_holder["angles"])

        mean_Lrel = float(jnp.mean(Lrel))
        mean_Lrot = float(jnp.mean(Lrot))
        mean_num  = float(jnp.mean(num_mean))
        mean_den  = float(jnp.mean(den_mean))

        hist_rel.append(mean_Lrel)
        hist_rot.append(mean_Lrot)
        hist_num.append(mean_num)
        hist_den.append(mean_den)

        if it_counter["it"] % print_every == 0:
            print(
                f"iter={it_counter['it']:4d} | "
                f"mean L_rel={mean_Lrel:.3e} | mean num={mean_num:.3e} | mean den={mean_den:.3e} | "
                f"mean L_rot={mean_Lrot:.3e}"
            )


        angles_holder["angles"] = sample_angles(rng_angles, K, N_steps, J_rot)
        it_counter["it"] += 1


    ka, kc = jax.random.split(jax.random.PRNGKey(1))
    a0 = jax.random.normal(ka, (s_stages, s_stages - 1)) * 0.1
    c0 = jax.random.normal(kc, (s_stages,)) * 0.1
    x0 = pack_thetas(a0, c0)


    Lrel0, Lrot0, num0, den0 = batch_components(y0s, hs, a0, c0, angles_holder["angles"])
    print(
        f"init | mean L_rel={float(jnp.mean(Lrel0)):.3e} | mean num={float(jnp.mean(num0)):.3e} "
        f"| mean den={float(jnp.mean(den0)):.3e} | mean L_rot={float(jnp.mean(Lrot0)):.3e}"
    )

    def obj(x_flat):
        a, c = unpack_thetas(x_flat, s_stages)
        val = batch_loss(y0s, hs, a, c, angles_holder["angles"], lambda_rot)
        vf = float(val)
        if not np.isfinite(vf):
            raise ValueError("Loss became non-finite.")
        return vf

    def grad(x_flat):
        g = jax.grad(lambda xf: batch_loss(y0s, hs, *unpack_thetas(xf, s_stages), angles_holder["angles"], lambda_rot))(
            jnp.array(x_flat)
        )
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
    a_star, c_star = unpack_thetas(res.x, s_stages)

    return a_star, c_star, (hist_rel, hist_rot, hist_num, hist_den), res



if __name__ == "__main__":
    K = 500
    s_stages = 3
    N_steps = 1
    J_rot = 5
    n_ref = 100

    y0s, hs = make_dataset(K=K, h_min=0.05, h_max=0.2, r_min=1.5, seed=0)

    a_star, c_star, (hist_rel, hist_rot, hist_num, hist_den), res = train_rknn_two_body(
        y0s=y0s,
        hs=hs,
        s_stages=s_stages,
        N_steps=N_steps,
        J_rot=J_rot,
        n_ref=n_ref,
        lambda_rot=1.0,
        tol=1e-6,
        maxiter=300,
        method="L-BFGS-B",
        angles_seed=42,
        print_every=1
    )

    print("theta_a:\n", np.array(a_star))
    print("theta_c:\n", np.array(c_star))


    its = np.arange(len(hist_rel))
    plt.figure()
    plt.plot(its, hist_rel, marker="o", label=r"$\overline{L}_{\mathrm{rel}}$")
    plt.plot(its, hist_rot, marker="o", label=r"$\overline{L}_{\mathrm{rot}}$")
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(its, hist_num, marker="o", label="mean num = ||y_nn - y_true||^2")
    plt.plot(its, hist_den, marker="o", label="mean den = ||y_heun - y_true||^2")
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Wert")
    plt.legend()
    plt.tight_layout()
    plt.show()

