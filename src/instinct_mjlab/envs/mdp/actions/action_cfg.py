from dataclasses import MISSING, dataclass

from mjlab.envs.mdp import JointPositionActionCfg
from mjlab.managers import ActionTerm, SceneEntityCfg

from . import joint_actions

@dataclass(kw_only=True)
class ActionOverridenJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for the action overridden delayed joint position action term.

    See :class:`ActionOverridenointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = joint_actions.ActionOverridenJointPositionAction

    asset_cfg: SceneEntityCfg = MISSING
    """Whether to override the action with the delayed action. Defaults to False."""

    override_value: float = 0.0
    """Delay in frames before the action is overridden. Defaults to 0."""
