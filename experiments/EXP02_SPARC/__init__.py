"""
EXP02 - SPARC Rotation Curve Transition Law

Claim: A deterministic transition radius r_* (collapse-threshold) is predicted 
from luminous profile only; one global α fits train set, holds on test.

Data: SPARC database (photometry + HI kinematics)
Acceptance: PASS if median ≤ 15% and ρ ≥ 0.6 on hold-out, no per-galaxy tuning
"""

from .preregistration import SPARCPreregistration, create_preregistration
from .data_loader import SPARCDataLoader
from .transition_predictor import TransitionPredictor
from .validation import run_validation

__all__ = [
    'SPARCPreregistration',
    'create_preregistration',
    'SPARCDataLoader',
    'TransitionPredictor',
    'run_validation'
]
