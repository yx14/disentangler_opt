import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# -----------------------------
# Setup
# -----------------------------
seed = 0
rng = np.random.default_rng(seed)

l = r = b = c = 4
rank = 12
n = l * r

B = rng.standard_normal((l * r, b * c))
B = B / np.linalg.norm(B, ord="fro")

# -----------------------------
# Objective
# -----------------------------
def obj(Q, B, l, r, b, c, rank):
    Y_unfold = Q @ B
    Y = Y_unfold.reshape((l, r, b, c), order="F")
    Yp = np.transpose(Y, (0, 2, 1, 3))
    M = Yp.reshape((l * b, r * c), order="F")
    s = np.linalg.svd(M, compute_uv=False)
    if rank >= s.size:
        return 0.0
    return float(np.sum(s[rank:] ** 2))

# -----------------------------
# Grid scan
# -----------------------------
N = 250
theta = np.linspace(-2 * np.pi, 2 * np.pi, N)
Z = np.zeros((N, N))

k = 1
v = np.zeros(n - k)
idx1, idx2 = 4, 5  # MATLAB v(5), v(6)

for i, s1 in enumerate(theta):
    for j, s2 in enumerate(theta):
        v[:] = 0.0
        v[idx1] = s1
        v[idx2] = s2
        S = np.diag(v, -k) + np.diag(-v, k)
        Q = expm(S)
        Z[i, j] = obj(Q, B, l, r, b, c, rank)

# -----------------------------
# Local minima detection
# -----------------------------
def local_minima_8nbr(Z):
    mins = []
    for i in range(1, Z.shape[0] - 1):
        for j in range(1, Z.shape[1] - 1):
            z = Z[i, j]
            nbrs = Z[i-1:i+2, j-1:j+2].copy()
            nbrs[1, 1] = np.inf
            if np.all(z < nbrs):
                mins.append((i, j))
    return mins

mins = local_minima_8nbr(Z)

if len(mins) == 0:
    raise RuntimeError("No strict local minima found. Try increasing N.")

mins_sorted = sorted(mins, key=lambda ij: Z[ij])
min_small = mins_sorted[0]
min_large = mins_sorted[-1]

picked = [("local min", min_small),
          ("local min", min_large)]

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(7, 6), facecolor="white")

cf = plt.contourf(theta, theta, Z, levels=60)
plt.colorbar(cf)

plt.xlabel(r"$s_1$", fontsize=16)
plt.ylabel(r"$s_2$", fontsize=16)
plt.title("objective", fontsize=18)

ax = plt.gca()
ax.tick_params(labelsize=12)
for spine in ax.spines.values():
    spine.set_linewidth(1)

# Mark and annotate extrema among local minima
for label, (i, j) in picked:
    s1, s2 = theta[i], theta[j]
    f = Z[i, j]

    # contourf convention: x=theta[j], y=theta[i]
    x_plot, y_plot = theta[j], theta[i]

    plt.plot(x_plot, y_plot, 'wo', markersize=8, markeredgewidth=2)

    plt.annotate(
        f"{label}\n"
        f"f={f:.3e}",
        xy=(x_plot, y_plot),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black"),
        arrowprops=dict(arrowstyle="->")
    )

plt.tight_layout()
plt.savefig('cost_landscape.eps')
plt.show()

# Print values
print(f"Found {len(mins)} strict local minima.")
for label, (i, j) in picked:
    s1, s2 = theta[i], theta[j]
    f = Z[i, j]
    print(f"{label}: (s1,s2)=({s1:.6f}, {s2:.6f}), f={f:.12e}")