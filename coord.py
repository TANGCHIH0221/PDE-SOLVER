import numpy as np

# ---------------------------
# Cartesian (x, y)
# ---------------------------
def make_grid_cartesian(nx, ny, dx, dy):
    x = np.linspace(0, (nx - 1) * dx, nx)
    y = np.linspace(0, (ny - 1) * dy, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    return X, Y

def laplacian_cartesian(Z, dx, dy):
    return ((np.roll(Z, -1, axis=0) - 2 * Z + np.roll(Z, 1, axis=0)) / dx**2 +
            (np.roll(Z, -1, axis=1) - 2 * Z + np.roll(Z, 1, axis=1)) / dy**2)

# ---------------------------
# Cylindrical
# ---------------------------
def make_grid_cylindrical(nr, nphi, dr, dphi):
    r = np.linspace(0, (nr - 1) * dr, nr)
    phi = np.linspace(0, (nphi - 1) * dphi, nphi)
    R, PHI = np.meshgrid(r, phi, indexing="ij")
    return R, PHI, r, phi

def laplacian_cylindrical(U, dr, dphi, r_array):
    nr, nphi = U.shape
    out = np.zeros_like(U)

    d2_phi = (np.roll(U, -1, axis=1) - 2 * U + np.roll(U, 1, axis=1)) / dphi**2

    # i = 0 (center)
    up = U[1, :]
    uc = U[0, :]
    d2_r0 = 2.0 * (up - uc) / dr**2  # ghost: u_-1 = u_1
    out[0, :] = d2_r0  # angular term at r=0 taken as finite → 0

    # i = 1..nr-1
    i = np.arange(1, nr)
    up = U[np.minimum(i + 1, nr - 1), :]
    um = U[i - 1, :]
    uc = U[i, :]
    r = r_array[i][:, None]
    d2_r = (up - 2 * uc + um) / dr**2 + (up - um) / (2 * dr) / r
    out[i, :] = d2_r + d2_phi[i, :] / (r**2 + 1e-15)
    return out

# ---------------------------
# Spherical 
# ---------------------------
def make_grid_spherical_radial(nr, dr):
    return np.linspace(0, (nr - 1) * dr, nr)

def laplacian_spherical_radial(u, dr, r_array):
    nr = u.shape[0]
    out = np.zeros_like(u)

    # r=0
    up = u[1]
    uc = u[0]
    out[0] = 2.0 * (up - uc) / dr**2  # ghost: u_-1 = u_1

    # r>0
    i = np.arange(1, nr)
    up = u[np.minimum(i + 1, nr - 1)]
    um = u[i - 1]
    uc = u[i]
    r = r_array[i]
    out[i] = (up - 2 * uc + um) / dr**2 + (2.0 / r) * (up - um) / (2 * dr)
    return out
