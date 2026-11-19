"""Quick-look figure: ground truth, observation, reconstruction(s) and error."""
import matplotlib.pyplot as plt
import numpy as np


def _np(x):
    x = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    return x.squeeze()


def show_reconstruction(gt, obs, recs, title=""):
    """``recs`` is a dict {label: 2-D reconstruction}. Error maps use the first reconstruction."""
    gt, obs = _np(gt), _np(obs)
    recs = {k: _np(v) for k, v in recs.items()}
    panels = [("Ground truth (0 m)", gt), ("Observation (100 m)", obs), *recs.items()]
    first = next(iter(recs.values()))
    panels.append((f"Error ({next(iter(recs))})", first - gt))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3))
    vmin, vmax = gt.min(), gt.max()
    for ax, (label, img) in zip(axes, panels):
        is_err = label.startswith("Error")
        im = ax.contourf(img, levels=30, cmap="RdBu_r" if is_err else "viridis",
                         **({} if is_err else {"vmin": vmin, "vmax": vmax}))
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(label, fontsize=9)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
