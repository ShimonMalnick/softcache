"""SoftCache, model-independent: the per-node decision, Residual Motion, and the jump regrid."""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Config:
    tau: float = 0.30      # refresh when the accumulated SeaCache rel-L1 reaches tau
    jf_max: float = 8.0    # cap on the jump factor
    beta: float = 1.0      # Residual Motion step; the paper uses 1.0 below ~4x speedup (tau ~0.3-0.5) and 0.5 above
    # jump gate (paper app:gate): no jump in the volatile early sigmas (sigma > 0.9) or when the
    # probe changes sharply at this node
    sigma_hi: float = 0.90
    raw_rel_l1_max: float = 0.18


def decide(cfg: Config, i: int, sigma: float, raw: float, acc: float) -> tuple[str, float | None]:
    """Return ("fresh", None), ("cache", None) or ("jump", jump_factor).

    Fresh at the first node and whenever the staleness budget is spent (SeaCache). Otherwise,
    if the gate allows, jump with a stride that grows with the unspent budget (headroom):
    jf = 1 + (jf_max - 1) * (1 - acc / tau)."""
    if i == 0 or acc >= cfg.tau:
        return "fresh", None
    if sigma <= cfg.sigma_hi and raw <= cfg.raw_rel_l1_max:
        headroom = max(0.0, 1.0 - acc / max(1e-9, cfg.tau))
        jf = 1.0 + (cfg.jf_max - 1.0) * headroom
        if jf > 1.01:      # a stride under 1.01 skips no scheduled node, so it is not a jump
            return "jump", jf
    return "cache", None


def moved_residual(r_anchor: torch.Tensor, r_prev: torch.Tensor, sigma: float,
                   sigma_anchor: float, sigma_prev: float, beta: float) -> torch.Tensor:
    """Residual Motion: extrapolate the block residual along the secant of the last two fresh
    anchors, r = r_a + beta * lam * (r_a - r_{a-1}), lam = (sigma - sigma_a) / (sigma_a - sigma_{a-1})."""
    # sigma decreases along the schedule, so the denominator is negative; the 1e-8 guard is
    # inherited from the paper runs and kept for bit-exactness, as is the (beta * lam) grouping.
    lam = (sigma - sigma_anchor) / (sigma_anchor - sigma_prev - 1e-8)
    return r_anchor + (beta * lam) * (r_anchor - r_prev)


def regrid_tail(sigmas: list[float], i: int, target: float) -> list[float]:
    """After jumping from node i to sigma=`target`, re-space the rest of the schedule linearly
    from target to 0 with only as many nodes as the old schedule had below target. The nodes
    this removes are the compute a jump saves."""
    n_tail = len(sigmas) - (i + 1)
    n_below = sum(1 for s in sigmas if 0.0 < s < target) + 1   # +1 for the terminal 0
    n_new = min(n_tail - 1, n_below)
    if n_new <= 1:   # no node left below target: still take the final step target -> 0
        return sigmas[: i + 1] + ([target, 0.0] if target > 0 else [0.0])
    tail = [target * (1.0 - k / (n_new - 1)) for k in range(n_new)]
    tail[0], tail[-1] = target, 0.0
    return sigmas[: i + 1] + tail


def euler(latents: torch.Tensor, dsigma: float, v: torch.Tensor) -> torch.Tensor:
    """Euler step, identical to diffusers' FlowMatchEulerDiscreteScheduler.step."""
    dt = torch.tensor(dsigma, dtype=torch.float32, device=v.device)
    return (latents.to(torch.float32) + dt * v).to(v.dtype)


def execution_device(pipe, module) -> torch.device:
    """Where `module` computes: its own device, or the GPU when diffusers' offload hooks park it
    on the CPU between calls."""
    return pipe._execution_device if hasattr(module, "_hf_hook") else module.device
