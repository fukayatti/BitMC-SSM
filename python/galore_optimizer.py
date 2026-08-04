"""
GaLore (Gradient Low-Rank Projection) Optimizer for PyTorch
Reference: Zhao et al. (2024) - Memory-Efficient LLM Training by Gradient Low-Rank Projection

Reduces AdamW optimizer state memory by 80%~90%+ on CPU/GPU by projecting 2D weight gradients
into a low-rank subspace (rank r << min(m, n)) and tracking moments only in the subspace.
"""

import math
import torch
from torch.optim.optimizer import Optimizer


class GaLoreAdamW(Optimizer):
    """
    GaLore-enhanced AdamW Optimizer.
    For 2D weight matrices (d_out, d_in), applies low-rank gradient projection.
    For 1D tensors (biases, norms, embeddings), applies standard AdamW.
    """
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        rank=16,
        update_proj_gap=200,
        scale=1.0
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            rank=rank,
            update_proj_gap=update_proj_gap,
            scale=scale
        )
        super(GaLoreAdamW, self).__init__(params, defaults)

    def _get_orthogonal_projection(self, grad: torch.Tensor, rank: int, proj_type: str = "left"):
        """Computes low-rank orthogonal projection matrix P via QR decomposition."""
        m, n = grad.shape
        if proj_type == "left":
            # Left projection: P is (m, r)
            if m < rank:
                rank = m
            # Fast randomized / standard QR
            q, _ = torch.linalg.qr(grad.float())
            return q[:, :rank]
        else:
            # Right projection: Q is (n, r)
            if n < rank:
                rank = n
            q, _ = torch.linalg.qr(grad.float().T)
            return q[:, :rank]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            rank = group['rank']
            update_proj_gap = group['update_proj_gap']
            scale = group['scale']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("GaLoreAdamW does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    is_2d = (p.ndim == 2 and min(p.shape) > rank)
                    state['is_galore'] = is_2d

                    if is_2d:
                        m, n = p.shape
                        # Choose left or right projection based on dimension
                        proj_type = "left" if m >= n else "right"
                        state['proj_type'] = proj_type
                        state['P'] = self._get_orthogonal_projection(grad, rank, proj_type)
                        
                        # Initialize low-rank moment buffers
                        if proj_type == "left":
                            # P is (m, r), projected grad is (r, n)
                            state['exp_avg'] = torch.zeros(rank, n, dtype=torch.float32, device=p.device)
                            state['exp_avg_sq'] = torch.zeros(rank, n, dtype=torch.float32, device=p.device)
                        else:
                            # Q is (n, r), projected grad is (m, r)
                            state['exp_avg'] = torch.zeros(m, rank, dtype=torch.float32, device=p.device)
                            state['exp_avg_sq'] = torch.zeros(m, rank, dtype=torch.float32, device=p.device)
                    else:
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state['step'] += 1
                step = state['step']

                # Perform weight decay
                if weight_decay != 0:
                    p.mul_(1.0 - lr * weight_decay)

                if state['is_galore']:
                    proj_type = state['proj_type']
                    # Periodic projection update
                    if step % update_proj_gap == 0:
                        state['P'] = self._get_orthogonal_projection(grad, rank, proj_type)
                        state['exp_avg'].zero_()
                        state['exp_avg_sq'].zero_()

                    P = state['P']
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']

                    # Project gradient into low-rank subspace
                    if proj_type == "left":
                        # (r, m) @ (m, n) -> (r, n)
                        proj_grad = torch.matmul(P.T, grad.float())
                    else:
                        # (m, n) @ (n, r) -> (m, r)
                        proj_grad = torch.matmul(grad.float(), P)

                    # Update low-rank Adam moments
                    exp_avg.mul_(beta1).add_(proj_grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(proj_grad, proj_grad, value=1.0 - beta2)

                    bias_correction1 = 1.0 - beta1 ** step
                    bias_correction2 = 1.0 - beta2 ** step

                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    step_size = lr / bias_correction1
                    norm_update = (exp_avg / denom) * scale

                    # Project back to full parameter space: delta_W = P @ norm_update
                    if proj_type == "left":
                        full_update = torch.matmul(P, norm_update)
                    else:
                        full_update = torch.matmul(norm_update, P.T)

                    p.add_(full_update.to(p.dtype), alpha=-step_size)

                else:
                    # Standard AdamW for 1D tensors
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']

                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                    bias_correction1 = 1.0 - beta1 ** step
                    bias_correction2 = 1.0 - beta2 ** step

                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    step_size = lr / bias_correction1
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
