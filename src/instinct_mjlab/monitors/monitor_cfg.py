from dataclasses import MISSING, dataclass

from mjlab.managers import ManagerTermBaseCfg

from .monitor_manager import MonitorTerm
from .monitors import TorqueMonitorSensor

@dataclass(kw_only=True)
class MonitorSensorCfg:
    class_type: type = None

    update_period: float = 0.005  # update every decimation

    entity_name: str = "robot"  # entity name to monitor

@dataclass(kw_only=True)
class MonitorTermCfg(ManagerTermBaseCfg):
    func: type[MonitorTerm] = MISSING

@dataclass(kw_only=True)
class TorqueMonitorSensorCfg(MonitorSensorCfg):
    """NOTE: Due to the update of joint_acc every decimation, it significantly decreases the performance (about 0.25x slower)."""

    class_type: type = TorqueMonitorSensor

    history_length: int = 4  # assuming is the number of decimation
