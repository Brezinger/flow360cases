@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PYTHON_EXE=%SCRIPT_DIR%..\..\venv\Scripts\python.exe"
set "CORRELATOR=%SCRIPT_DIR%..\..\mesh_utils\correlate_gmsh_entities.py"

"%PYTHON_EXE%" "%CORRELATOR%" ^
  "%SCRIPT_DIR%msh_def_XWing2_2_rect.json" ^
  "%SCRIPT_DIR%msh_def_XWing2_2_rect_shifted_3_8mm.json" ^
  --entity-shift 3.8 35 ^
  --entity-shift 3.8 -35 ^
  --entity-shift 3.8 145 ^
  --entity-shift 3.8 215

pause
