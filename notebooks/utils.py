import os
import numpy as np
import ipywidgets as widgets
import json

import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D


class ParticleSimulation:
    def __init__(
        self,
        n=20,
        d=3,
        beta=0.0,
        dt=0.01,
        steps=1000,
        seed=0,
        antipodal_pair=False,
        output_dir="../plots",
    ):
        self.n = n
        self.d = d
        self.beta = beta
        self.dt = dt
        self.steps = steps
        self.seed = seed
        self.antipodal_pair = antipodal_pair
        self.trajectory = None

        # Directory based on config
        self.save_dir = os.path.join(
            output_dir, f"d={self.d}_beta={self.beta}_n={self.n}"
        )
        os.makedirs(self.save_dir, exist_ok=True)
        # Save config
        config = {
            "d": self.d,
            "beta": self.beta,
            "n": self.n,
            "dt": self.dt,
            "steps": self.steps,
            "seed": self.seed,
            "antipodal_pair": self.antipodal_pair,
        }
        with open(os.path.join(self.save_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def compute_trajectory(self):
        if self.d < 2:
            raise ValueError("Dimension d must be >= 2.")

        np.random.seed(self.seed)
        x = np.random.randn(self.n, self.d)
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        if self.antipodal_pair:
            base = np.random.randn(self.d)
            base = base / np.linalg.norm(base)
            x[0] = base
            x[1] = -base
            x = x[:2]
            self.n = 2

        def project(xi, y):
            return y - np.dot(xi, y) * xi

        trajectory = np.zeros((self.steps, self.n, self.d))
        trajectory[0] = x.copy()
        for step in range(1, self.steps):
            x_new = np.zeros((self.n, self.d))
            for i in range(self.n):
                dots = np.dot(x[i], x.T)
                weights = np.exp(self.beta * dots)
                Z = np.sum(weights)
                if not np.isfinite(Z) or Z == 0:
                    raise ValueError(f"Z={Z} at step={step}, particle={i}")
                weighted_sum = np.sum(weights[:, np.newaxis] * x, axis=0) / Z
                delta = project(x[i], weighted_sum)
                xi_new = x[i] + self.dt * delta
                x_new[i] = xi_new / np.linalg.norm(xi_new)
            x = x_new
            trajectory[step] = x
        self.trajectory = trajectory

    def _make_title(self, step, plot_dim):
        proj_info = f" | {plot_dim}D projection" if self.d > plot_dim else ""
        return f"Step {step} | n={self.n}, d={self.d}, β={self.beta}{proj_info}"

    def plot_step(
        self,
        step,
        ax=None,
        force_2d=False,
        figsize=(6, 6),
        show=True,
        save=False,
        fname=None,
        trace_length=0,
        trace_color="blue",
        trace_alpha=0.75,
        trace_gamma=4,
        trace_lw=1,
        azim=45,
        elev=25,
        drop_title=False,
        margin_3d=-0.2,
    ):
        if self.trajectory is None:
            raise ValueError("Run compute_trajectory() first.")
        x_step = self.trajectory[step]
        use_3d = self.d >= 3 and not force_2d
        plot_dim = 3 if use_3d else 2

        trace_lw -= 0.2 if use_3d else 0

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d" if use_3d else None)
        ax.clear()

        if use_3d:
            ax.scatter(x_step[:, 0], x_step[:, 1], x_step[:, 2], color="blue", s=50)
            u, v = np.linspace(0, 2 * np.pi, 30), np.linspace(0, np.pi, 15)
            xs = np.outer(np.cos(u), np.sin(v))
            ys = np.outer(np.sin(u), np.sin(v))
            zs = np.outer(np.ones_like(u), np.cos(v))

            ax.plot_wireframe(xs, ys, zs, color="gray", alpha=0.2, linewidth=0.2)

            ax.set_xlim([-1 - margin_3d, 1 + margin_3d])
            ax.set_ylim([-1 - margin_3d, 1 + margin_3d])
            ax.set_zlim([-1 - margin_3d, 1 + margin_3d])
            ax.set_box_aspect([1, 1, 1])
            ax.view_init(elev=elev, azim=azim)
            ax.grid(False)

        else:
            ax.add_artist(
                plt.Circle((0, 0), 1, color="gray", fill=False, alpha=0.3, lw=0.2)
            )
            ax.set_aspect("equal")
            if self.d > 2:
                ax.scatter(
                    x_step[:, 0], x_step[:, 1], c=x_step[:, 2], cmap="coolwarm", s=50
                )
            else:
                ax.scatter(x_step[:, 0], x_step[:, 1], color="blue", s=50)
            ax.set_xlim([-1.2, 1.2])
            ax.set_ylim([-1.2, 1.2])

        # if trace_length > 0 plot the trace of the particle
        if trace_length > 0:
            start = max(0, step - trace_length)
            for i in range(self.n):
                traj = self.trajectory[start : step + 1, i, :]
                for j in range(1, len(traj)):
                    alpha = trace_alpha * (j / (len(traj) - 1)) ** trace_gamma
                    if use_3d:
                        ax.plot(
                            traj[j - 1 : j + 1, 0],
                            traj[j - 1 : j + 1, 1],
                            traj[j - 1 : j + 1, 2],
                            color=trace_color,
                            linewidth=trace_lw,
                            alpha=alpha,
                        )
                    else:
                        ax.plot(
                            traj[j - 1 : j + 1, 0],
                            traj[j - 1 : j + 1, 1],
                            color=trace_color,
                            linewidth=trace_lw,
                            alpha=alpha,
                        )
        
        ax.set_axis_off()
        if not drop_title:
            ax.set_title(self._make_title(step, plot_dim), pad=0)

        if save:
            if fname is None:
                extra_info = f"_force2d={force_2d}" if force_2d else ""
                extra_info += f"_drop_title={drop_title}" if drop_title else ""
                fname = f"step={step}_d={self.d}_beta={self.beta}_n={self.n}_steps={self.steps}{extra_info}.pdf"
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, fname))

        if show:
            plt.tight_layout()
            plt.show()

        else:
            return ax

    def plot_selected_steps(
        self,
        plt_steps=None,
        figsize=(15, 5),
        force_2d=False,
        save=False,
        fname=None,
        trace_length=0,
        azim=45,
        elev=25,
    ):
        if self.trajectory is None:
            raise ValueError("Run compute_trajectory() first.")
        if plt_steps is None:
            plt_steps = [0, self.steps // 2, self.steps - 1]
        use_3d = self.d >= 3 and not force_2d
        fig, axs = plt.subplots(
            1,
            len(plt_steps),
            figsize=figsize,
            subplot_kw={"projection": "3d"} if use_3d else {},
        )
        if len(plt_steps) == 1:
            axs = [axs]
        for ax, step in zip(axs, plt_steps):
            self.plot_step(
                step,
                ax=ax,
                force_2d=force_2d,
                show=False,
                trace_length=trace_length,
                azim=azim,
                elev=elev,
            )
        plt.tight_layout()

        if save:
            if fname is None:
                extra_info = f"_force2d={force_2d}" if force_2d else ""
                fname = f"progression_d={self.d}_beta={self.beta}_n={self.n}_steps={self.steps}{extra_info}.pdf"
            plt.savefig(os.path.join(self.save_dir, fname), bbox_inches="tight")

        plt.show()

    def generate_animation(self, fname=None, fps=15, force_2d=False, trace_length=0):
        if self.trajectory is None:
            raise ValueError("Run compute_trajectory() first.")

        if fname is None:
            extra_info = f"_force2d={force_2d}" if force_2d else ""
            fname = f"anim_d={self.d}_beta={self.beta}_n={self.n}_steps={self.steps}{extra_info}.mp4"
        filepath = os.path.join(self.save_dir, fname)

        fig = plt.figure(figsize=(6, 6))
        ax = (
            fig.add_subplot(111, projection="3d")
            if self.d >= 3 and not force_2d
            else fig.add_subplot(111)
        )

        def update(frame):
            self.plot_step(frame, ax=ax, show=False, trace_length=trace_length)
            return (ax,)

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=range(0, self.steps, max(1, self.steps // 100)),
            blit=False,
            repeat=False,
        )

        ani.save(filepath, writer="ffmpeg", fps=fps, dpi=200)

        plt.close()

    def interactive_widget(self, figsize=(6, 6)):
        if self.trajectory is None:
            raise ValueError("Run compute_trajectory() first.")

        def plot_step(step):
            self.plot_step(step, ax=None, force_2d=False, figsize=figsize)

        slider = widgets.IntSlider(
            min=0, max=self.steps - 1, step=10, value=0, description="Step"
        )
        widgets.interact(plot_step, step=slider)
