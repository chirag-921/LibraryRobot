# LibraryRobot

LLM-guided autonomous mobile librarian robot (MuJoCo, Franka Panda on a wheeled base).
Plan: see `project_phases_mobile_mujoco.md`.

## Setup (Windows, PowerShell)

```powershell
git clone https://github.com/chirag-921/LibraryRobot.git
cd LibraryRobot
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
git clone --branch v2026.11.2 --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git assets/mujoco_menagerie
pytest tests
```

Known issue: Windows Application Control can block `mujoco.dll` (WinError 4551).
XML generation and the structural test work without it; scripts that run MuJoCo need it.
