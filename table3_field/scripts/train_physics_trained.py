"""Fit the Physics-trained network (Sun et al., 2025) to the 10 field observations.

The method has no training set: a freshly initialised network is fitted directly to the test observations
y = A x through the physics loss ||A f(y) - y||_1, where A is the single 800 m upward continuation.
The ground truth is never used for fitting; it is only read to log the metrics during the fit.

Recipe (Physics_trained_net.ipynb): DownContinuationNet (ConvNeXt U-Net without time embedding), Adam,
learning rate 2e-5, L1 physics loss. Here all 10 observations form one batch, so one epoch is one update;
the released weights are 10,000 updates with seed 0 (the defaults).
Full FP32 with deterministic cuDNN and a fixed seed, so the fit is repeatable on the same hardware/software.

Usage:  python scripts/train_physics_trained.py            # seed 0, 10,000 updates -> weights/physics_trained_seed0.pth
        python scripts/train_physics_trained.py --seed 1 [--epochs 10000] [--log-every 100]
After saving, the refit is compared tensor by tensor with the released weights/physics_trained.pth (if present).
"""
import argparse
import csv
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from gravdc import evaluate, load_test_set, summarize, upward_continue  # noqa: E402  (FP32 set on import)
from gravdc.models.unet import DownContinuationNet  # noqa: E402
from gravdc.physics import apply_kernel, continuation_kernel, padded_size  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=10000)
p.add_argument("--lr", type=float, default=2e-5)
p.add_argument("--log-every", type=int, default=100)
p.add_argument("--out", default=None, help="weight file (default: weights/physics_trained_seed<seed>.pth)")
p.add_argument("--log", default=None, help="CSV log (default: results/physics_fit_seed<seed>.csv)")
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
out = args.out or os.path.join(ROOT, "weights", f"physics_trained_seed{args.seed}.pth")
log = args.log or os.path.join(ROOT, "results", f"physics_fit_seed{args.seed}.csv")
os.makedirs(os.path.dirname(out), exist_ok=True)
os.makedirs(os.path.dirname(log), exist_ok=True)

names, x = load_test_set(os.path.join(ROOT, "data", "test"))
x = x.to(device)
y = upward_continue(x, 800)                                   # observations (800 m, zero padding to 36 x 36)
kernel = torch.as_tensor(continuation_kernel(800, padded_size(32, 32)), dtype=torch.float32, device=device)


def A(z):  # differentiable forward operator (same filter as upward_continue)
    return apply_kernel(z, kernel)


torch.manual_seed(args.seed)
np.random.seed(args.seed)
model = DownContinuationNet().to(device).train()
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
l1 = torch.nn.L1Loss()

t0 = time.perf_counter()
with open(log, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["epoch", "physics_loss", "PSNR", "PSNR_std", "SSIM", "SSIM_std", "RE", "RE_std", "RMSE", "RMSE_std"])
    for epoch in range(1, args.epochs + 1):
        loss = l1(A(model(y)), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % args.log_every == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                x_hat = model(y)
                phys = l1(A(x_hat), y).item()
            s = summarize(evaluate(x, x_hat))                   # ground truth used for logging only
            model.train()
            w.writerow([epoch, phys] + [v for k in ("PSNR", "SSIM", "RE", "RMSE") for v in s[k]])
            f.flush()
            print(f"seed {args.seed}  epoch {epoch:6d}  physics loss {phys:.6f}  PSNR {s['PSNR'][0]:.4f} ± {s['PSNR'][1]:.4f}"
                  f"  ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)

state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}  # CPU tensors: device-independent file
torch.save(state, out)
print(f"saved {out}  ({(time.perf_counter() - t0) / 60:.1f} min)")

released = os.path.join(ROOT, "weights", "physics_trained.pth")
if os.path.exists(released) and os.path.abspath(released) != os.path.abspath(out):
    ref = torch.load(released, map_location="cpu", weights_only=True)
    same = list(ref) == list(state) and all(torch.equal(ref[k], state[k]) for k in ref)
    diff = max((ref[k] - state[k]).abs().max().item() for k in ref) if list(ref) == list(state) else float("nan")
    print(f"identical to weights/physics_trained.pth: {same}  (max |difference| {diff:.3g})")
