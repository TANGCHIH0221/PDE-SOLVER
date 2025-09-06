import argparse
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from .coords import make_grid_cartesian, laplacian_cartesian
from .utils import apply_boundary, cfl_dt
from .spectral_fourier import laplacian_fft

def simulate_wave(nx=200, ny=200, dx=1.0, dy=1.0, c=1.0,
                  steps=150, bc_type="neumann",
                  method="fdm", save_gif=False, gif_path="figures/wave.gif"):
    """
    2D wave eq: u_tt = c^2 ∇^2 u
    Explicit leapfrog-like update:
        U_next = 2U - U_prev + (c*dt)^2 * Lap(U)
    """
    X, Y = make_grid_cartesian(nx, ny, dx, dy)

    # initial condition: Gaussian pulse
    x0, y0, sigma = 0.5 * (nx-1) * dx, 0.5 * (ny-1) * dy, 10.0
    U = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
    U_prev = U.copy()  # zero initial velocity

    dt = cfl_dt(c, dx, dy)
    frames = []

    for step in range(steps):
        if method == "fdm":
            lap = laplacian_cartesian(U, dx, dy)
        elif method == "fft":
            if bc_type != "periodic":
                raise ValueError("Spectral method requires periodic BC.")
            lap = laplacian_fft(U, dx, dy)
        else:
            raise ValueError("method must be 'fdm' or 'fft'")

        U_next = 2 * U - U_prev + (c * dt)**2 * lap
        U_next = apply_boundary(U_next, bc_type)

        # rotate time levels
        U_prev, U = U, U_next

        # capture frames sparsely
        if save_gif and (step % 2 == 0):
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(U, origin="lower", cmap="viridis")
            ax.set_title(f"t = {step*dt:.2f}")
            ax.set_xticks([]); ax.set_yticks([])
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(frame)
            plt.close(fig)

    if save_gif:
        imageio.mimsave(gif_path, frames, fps=20)
        print(f"[Saved] {gif_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--nx", type=int, default=200)
    p.add_argument("--ny", type=int, default=200)
    p.add_argument("--dx", type=float, default=1.0)
    p.add_argument("--dy", type=float, default=1.0)
    p.add_argument("--c", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--bc", type=str, default="neumann", choices=["dirichlet", "neumann", "periodic"])
    p.add_argument("--method", type=str, default="fdm", choices=["fdm", "fft"])
    p.add_argument("--gif", type=str, default="")
    args = p.parse_args()

    gif_path = args.gif if args.gif else ("figures/wave_2d_%s.gif" % args.bc)
    simulate_wave(nx=args.nx, ny=args.ny, dx=args.dx, dy=args.dy, c=args.c,
                  steps=args.steps, bc_type=args.bc,
                  method=args.method, save_gif=True, gif_path=gif_path)
