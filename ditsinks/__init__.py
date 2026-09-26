"""High-norm tokens, attention sinks and massive activation channels in DiTs.

One analysis, three model families (FLUX.1, FLUX.2, PixArt-Sigma).
"""
from .config import SweepConfig, DEFAULT_PROMPTS
from .registry import MODEL_REGISTRY, ModelSpec, registry_table, resolve_model
from .runner import SweepResult, run_sweep
from .causal_records import (CausalResults, CleanTargetManifest,
                             IncompatibleCausalRun, InterventionRecord,
                             LayerHeadMeasurement, ResumeIdentity, TokenGeometry)
from .causal_engine import (CausalTracer, EditPlan, FrozenTargets, GenerationDriver, Trace,
                            run_traced_generation, select_frozen_targets)
from .questions import (QUESTION_RUNNERS, QUESTION_TITLES, QuestionContext, QuestionResult,
                        run_diagnostic_harness, run_direction_magnitude_gate, run_questions)
from .q11 import build_schemes, lifecycle_windows, run_q11
from .control_surface import (ControlSurfaceConfig, ControlSurfaceResult, align_scale,
                              control_surface_states, gap_closed, identity_error,
                              run_control_surface, select_treatment)
from .geometry import (Basis, GeometrySnapshot, build_basis, capture_snapshots,
                       categorise, exact_token_basis, interpretable_basis, pca_basis,
                       token_table)
from .provenance import create_run_layout

__all__ = [
    "SweepConfig",
    "DEFAULT_PROMPTS",
    "MODEL_REGISTRY",
    "ModelSpec",
    "registry_table",
    "resolve_model",
    "SweepResult",
    "run_sweep",
    "CausalResults",
    "CleanTargetManifest",
    "IncompatibleCausalRun",
    "InterventionRecord",
    "LayerHeadMeasurement",
    "ResumeIdentity",
    "TokenGeometry",
    "CausalTracer",
    "EditPlan",
    "FrozenTargets",
    "GenerationDriver",
    "Trace",
    "run_traced_generation",
    "select_frozen_targets",
    "QUESTION_RUNNERS",
    "QUESTION_TITLES",
    "QuestionContext",
    "QuestionResult",
    "run_questions",
    "run_q11",
    "build_schemes",
    "lifecycle_windows",
    "run_control_surface",
    "ControlSurfaceConfig",
    "ControlSurfaceResult",
    "align_scale",
    "control_surface_states",
    "identity_error",
    "select_treatment",
    "gap_closed",
    "Basis",
    "GeometrySnapshot",
    "build_basis",
    "capture_snapshots",
    "categorise",
    "exact_token_basis",
    "interpretable_basis",
    "pca_basis",
    "token_table",
    "run_diagnostic_harness",
    "run_direction_magnitude_gate",
    "create_run_layout",
]
__version__ = "0.1.0"
