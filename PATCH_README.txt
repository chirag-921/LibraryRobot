LibraryRobot — MuJoCo import/App Control compatibility patch

Problem addressed
The model generator imported mujoco at module import time. On Windows,
Application Control may reject mujoco.dll and Python raises OSError
(WinError 4551), not ImportError. This prevented even XML-only generation.

Change
- Catch ImportError and OSError during optional MuJoCo import.
- Preserve the original error and include it in a clear error if a caller
  requests MuJoCo runtime functionality.
- XML generation remains available without loading the MuJoCo DLL.

Apply
Extract this ZIP into C:\Robotics\LibraryRobot and replace the one file: 
  src\library_robot\sim\mobile_model.py

Then run:
  .\.venv\Scripts\Activate.ps1
  python scripts\10_generate_mobile_models.py

This patch does NOT bypass, disable, modify, or weaken Windows security.
It only separates XML generation from MuJoCo runtime loading.
The following still require the DLL to load and will remain blocked until
the Windows policy issue is resolved:
  python scripts\11_test_mobile_models.py
  python scripts\12_open_mobile_robot.py

After XML generation, inspect the Windows CodeIntegrity event (usually 3077)
and the DLL signature/hash to identify the active policy and blocked file.
