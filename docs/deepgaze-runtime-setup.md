# DeepGaze Runtime Setup

Use the dedicated setup script when you want a reproducible DeepGaze runtime on a new Windows machine.

## One-click entry

```powershell
scripts\setup-deepgaze-runtime.cmd
```

The script will:

1. Check Python 3.12 availability.
2. Check the Microsoft Visual C++ x64 runtime and upgrade it if the version is lower than the known-good minimum.
3. Create or reuse `.deepgaze-py312`.
4. Install the base project package.
5. Install pinned DeepGaze runtime dependencies from `configs/deepgaze-runtime-requirements.txt`.
6. Run `probe_deepgaze_runtime()` as a smoke test.

## Optional flags

```powershell
scripts\setup-deepgaze-runtime.cmd -ForceRecreate
scripts\setup-deepgaze-runtime.cmd -RunFullValidation
scripts\setup-deepgaze-runtime.cmd -SkipVcRuntime
scripts\setup-deepgaze-runtime.cmd -SkipSmokeTest
scripts\setup-deepgaze-runtime.cmd -VenvPath C:\runtime\deepgaze312
```

## Notes

- The script targets the same `.deepgaze-py312` runtime that the dashboard and worker already know how to use.
- The Visual C++ runtime upgrade is included because the known `torch/c10.dll` failure on this project was caused by an outdated `msvcp140.dll`.
- The pinned versions in `configs/deepgaze-runtime-requirements.txt` are the combinations that were actually validated in this repository.
- `-RunFullValidation` performs two real cognitive saliency inference checks: one without fixation history to validate `DeepGazeIIE`, and one with fixation history to validate `DeepGazeIII`.
