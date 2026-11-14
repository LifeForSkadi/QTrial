@echo off
rem ============================================================
rem  QTrial environment setup (Windows)
rem  usage:   setup.bat          -> core deps only (judge-ready)
rem           setup.bat full     -> core + qiskit/pytket/etc
rem ============================================================
setlocal
cd /d "%~dp0\.."

where python >nul 2>nul
if errorlevel 1 (
    echo [x] python not found in PATH.
    echo     Install Python 3.10-3.13 from https://www.python.org/downloads/
    echo     and make sure "Add python.exe to PATH" is checked.
    exit /b 1
)

if not exist .venv (
    echo [*] creating virtual environment .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo [x] venv creation failed.
        exit /b 1
    )
) else (
    echo [*] .venv already exists, reusing it
)

call .venv\Scripts\activate.bat

echo [*] upgrading pip ...
python -m pip install --upgrade pip -q

echo [*] installing core deps: torch numpy networkx pyyaml numba ...
pip install torch numpy networkx pyyaml numba -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 (
    echo [!] tsinghua mirror failed, retrying with default index ...
    pip install torch numpy networkx pyyaml numba
    if errorlevel 1 (
        echo [x] core deps install failed.
        exit /b 1
    )
)

if /i "%1"=="full" (
    echo [*] installing full deps (qiskit/pytket/mqt.bench/fastapi/cqlib) ...
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    if errorlevel 1 (
        echo [!] tsinghua mirror failed, retrying with default index ...
        pip install -r requirements.txt
        if errorlevel 1 (
            echo [x] full deps install failed (core deps are already installed).
            exit /b 1
        )
    )
) else (
    echo [*] core-only mode. For benchmark baselines run: setup.bat full
)

echo [*] smoke test: mapping examples\ghz10.qasm ...
python -m qtrail.pure_cli examples\ghz10.qasm >nul 2>nul
if errorlevel 1 (
    echo [!] smoke test failed - see error above, but deps may still be fine.
    echo     try: .venv\Scripts\python -m qtrail.pure_cli examples\ghz10.qasm
) else (
    echo [*] smoke test passed. outputs written to out\
)

echo.
echo [*] done. quick start:
echo     .venv\Scripts\activate
echo     python -m qtrail.pure_cli examples\qft5.qasm
echo.
echo     note: default pip wheel is CPU-only torch. If you have an NVIDIA GPU
echo     and want CUDA speedup, install torch from https://pytorch.org/get-started
echo     (CUDA is optional - the model is tiny, CPU is fine).
endlocal
