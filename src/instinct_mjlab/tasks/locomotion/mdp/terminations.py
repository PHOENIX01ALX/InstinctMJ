# Copyright (c) 2022-2025, The Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define terminations for locomotion."""

from __future__ import annotations

import torch

from mjlab.sensor import ContactSensor


def illegal_contact(
  env,
  sensor_name: str,
  threshold: float = 1.0,
) -> torch.Tensor:
  """Terminate when contact force exceeds threshold."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  return torch.any(
    torch.max(torch.linalg.vector_norm(contact_sensor.data.force_history, dim=-1), dim=2)[0] > threshold,
    dim=1,
  )
