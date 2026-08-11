"""Unit 10 starter: add degree-2 spherical-harmonic appearance to the full trainer."""

from __future__ import annotations


class SHTrainer:
    """TODO: extend the Unit 9 trainer with view-dependent RGB coefficients."""

    def __init__(self, base_trainer):
        raise NotImplementedError

    def render_and_backward(self, view_id: int):
        """TODO: evaluate SH colour and send its gradient through compositing."""
        raise NotImplementedError
