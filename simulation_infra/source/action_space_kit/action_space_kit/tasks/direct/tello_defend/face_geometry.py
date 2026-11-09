"""The breach face: one face of a cuboid that attackers must fly through. Pure torch.

Env-local frame: the face is the plane x = 0, spanning y in [-width/2, width/2] and z in
[bottom, bottom + height]; the box is behind it (x < 0) and attackers come from +x.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Face:
    width: float = 3.0
    height: float = 2.0
    bottom: float = 0.5

    def distance(self, pos: torch.Tensor) -> torch.Tensor:
        """Distance from points (..., 3) to the face rectangle, (...)."""
        dx = pos[..., 0].clamp(min=0.0)
        dy = (pos[..., 1].abs() - 0.5 * self.width).clamp(min=0.0)
        dz = torch.maximum(self.bottom - pos[..., 2], pos[..., 2] - (self.bottom + self.height)).clamp(min=0.0)
        return torch.sqrt(dx * dx + dy * dy + dz * dz)

    def inside(self, pos: torch.Tensor) -> torch.Tensor:
        """Whether points (..., 3) project inside the face rectangle (y, z), (...) bool."""
        return (pos[..., 1].abs() <= 0.5 * self.width) & (pos[..., 2] >= self.bottom) & (
            pos[..., 2] <= self.bottom + self.height)

    def grid(self, ny: int, nz: int, device) -> torch.Tensor:
        """(ny * nz, 3) points covering the face, edges and corners included."""
        y = torch.linspace(-0.5 * self.width, 0.5 * self.width, ny, device=device)
        z = torch.linspace(self.bottom, self.bottom + self.height, nz, device=device)
        yy, zz = torch.meshgrid(y, z, indexing="ij")
        return torch.stack([torch.zeros_like(yy), yy, zz], -1).reshape(-1, 3)

    def margin(self, d_pos: torch.Tensor, a_pos: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
        """Equal-speed guarding margin: min over face points T of |A - T| - |D - T|.

        Positive: the defender reaches every point of the face before the attacker (Isaacs'
        target guarding with equal speeds), so that attacker cannot breach past it. Shapes
        broadcast: d_pos (..., 3) and a_pos (..., 3) -> (...).
        """
        da = torch.linalg.norm(a_pos.unsqueeze(-2) - pts, dim=-1)
        dd = torch.linalg.norm(d_pos.unsqueeze(-2) - pts, dim=-1)
        return (da - dd).min(dim=-1).values
