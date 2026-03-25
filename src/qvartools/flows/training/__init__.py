"""training --- Flow training objectives and utilities."""

from __future__ import annotations

from qvartools.flows.training.flow_nqs_training import (
    FlowNQSConfig,
    create_physics_guided_trainer,
)
from qvartools.flows.training.flow_nqs_training import (
    PhysicsGuidedFlowTrainer as FlowNQSTrainer,
)
from qvartools.flows.training.loss_functions import (
    compute_entropy_loss,
    compute_local_energy,
    compute_physics_loss,
    compute_teacher_loss,
)
from qvartools.flows.training.physics_guided_training import (
    PhysicsGuidedConfig,
    PhysicsGuidedFlowTrainer,
)

__all__ = [
    "PhysicsGuidedConfig",
    "PhysicsGuidedFlowTrainer",
    "compute_teacher_loss",
    "compute_physics_loss",
    "compute_entropy_loss",
    "compute_local_energy",
    "FlowNQSConfig",
    "FlowNQSTrainer",
    "create_physics_guided_trainer",
]
