"""List available Instinct-RL tasks."""

from __future__ import annotations

import tyro
from prettytable import PrettyTable

import instinct_mjlab.tasks  # noqa: F401
import mjlab
from instinct_mjlab.tasks.registry import list_tasks


def list_environments(keyword: str | None = None) -> int:
  table = PrettyTable(["#", "Task ID"])
  table.title = "Available Environments in Instinct_mjlab"
  table.align["Task ID"] = "l"

  idx = 0
  for task_id in list_tasks():
    if keyword and keyword.lower() not in task_id.lower():
      continue
    idx += 1
    table.add_row([idx, task_id])

  print(table)
  if idx == 0:
    msg = "[INFO] No tasks matched"
    if keyword:
      msg += f" keyword '{keyword}'"
    print(msg)
  return idx


def main() -> None:
  tyro.cli(list_environments, config=mjlab.TYRO_FLAGS)


if __name__ == "__main__":
  main()
