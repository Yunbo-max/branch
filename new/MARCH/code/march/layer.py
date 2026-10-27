"""One MARCH layer: unchanged GDN recurrence + content-routed anchor readout."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import MARCHConfig
from .mlp import GatedMLP
from .recurrence import GatedDeltaRuleReference
from .routing import historical_readout, routing_weights


class MARCHLayer(nn.Module):
    def __init__(self, config: MARCHConfig) -> None:
        super().__init__()
        self.config = config
        self.recurrence = GatedDeltaRuleReference(
            config.hidden_size, config.num_heads, config.head_dim
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Anchor-position projections (Eq. 10-11).
        self.anchor_q_proj = nn.Linear(
            config.hidden_size, config.num_heads * config.head_dim, bias=False
        )
        self.anchor_k_proj = nn.Linear(config.hidden_size, config.routing_dim, bias=False)
        # Text-token routing (Eq. 12-13).
        self.route_proj = nn.Linear(config.hidden_size, config.routing_dim, bias=False)
        self.null_proj = nn.Linear(config.hidden_size, 1, bias=True)
        self.out_proj = nn.Linear(config.num_heads * config.head_dim, config.hidden_size, bias=False)
        # Standard transformer block tail, mirroring the GDN baseline block.
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = GatedMLP(config.hidden_size, intermediate_size=getattr(config, "intermediate_size", None))

    def forward(
        self,
        h: Tensor,              # (B, L_exp, D)
        text_mask: Tensor,      # (B, L_exp) bool
        anchor_pos: Tensor,     # (B, M) long
        boundaries: Tensor,     # (B, M) long
    ) -> Tensor:
        cfg = self.config
        b, l_exp, d = h.shape
        t = int(text_mask[0].sum().item())          # no left-padding in this reference
        m = anchor_pos.shape[1]
        nh, dk = cfg.num_heads, cfg.head_dim
        dv = dk                                      # expand_v = 1 in the reference

        h_text = h[text_mask].view(b, t, d)         # (B, T, D) in sequence order
        o_cur, states = self.recurrence(h_text)     # (B,T,H,dv), (B,T,H,dv,dk)

        # --- state anchors: snapshot the *cumulative* state at each boundary ---
        # A^(m) = S_{b_m - 1}; boundary b_m = m*C counts text tokens before the anchor.
        idx = (boundaries - 1).clamp(min=0)                     # (B, M)
        st = states.permute(0, 2, 1, 3, 4)                      # (B, H, T, dv, dk)
        gi = idx.view(b, 1, m, 1, 1).expand(b, nh, m, dv, dk)
        anchors = torch.gather(st, 2, gi).permute(0, 2, 1, 3, 4)  # (B, M, H, dv, dk)
        anchors = anchors * (boundaries > 0).view(b, m, 1, 1, 1)

        # --- anchor positions: read own checkpoint, emit routing key (Eq. 10-11) ---
        h_anchor = torch.gather(
            h, 1, anchor_pos.clamp(min=0).unsqueeze(-1).expand(b, m, d)
        )  # (B, M, D)
        u = self.norm(h_anchor)
        q_anchor = self.anchor_q_proj(u).view(b, m, nh, dk)
        o_anchor = torch.einsum("bmhvk,bmhk->bmhv", anchors, q_anchor).reshape(b, m, nh * dk)
        anchor_out = self.out_proj(o_anchor) + h_anchor
        kappa = self.anchor_k_proj(u)               # (B, M, d_r) content-conditioned keys

        # --- text routing (Eq. 12-13) ---
        x = self.norm(h_text)
        route_query = self.route_proj(x)
        null_logit = self.null_proj(x).squeeze(-1)  # (B, T)
        text_order = torch.arange(t, device=h.device).view(1, t, 1)
        visible = boundaries.view(b, 1, m) <= text_order  # b_m <= t
        pi = routing_weights(route_query, kappa, null_logit, visible, cfg.top_k)

        q_text = self.recurrence.q_proj(h_text).view(b, t, nh, dk)
        o_hist = historical_readout(pi, anchors, q_text)  # (B, T, H, dv)
        text_out = self.out_proj((o_cur + o_hist).reshape(b, t, nh * dk)) + h_text

        out = h.clone()
        out[text_mask] = text_out.reshape(-1, d)
        out.scatter_(1, anchor_pos.clamp(min=0).unsqueeze(-1).expand(b, m, d), anchor_out)
        # FFN tail (applies to text and anchor positions alike).
        return out + self.mlp(self.norm2(out))
