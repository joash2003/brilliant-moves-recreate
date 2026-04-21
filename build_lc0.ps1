# build_lc0.ps1 -- build the patched lc0 engine on Windows
# Uses VS 2022 BuildTools + CUDA 12.x + cuDNN 9.x
# Backends: plain CUDA + cuDNN (all others disabled)
#
# Portability: paths below can be overridden via environment variables
# so a grader does not have to edit this file. Example:
#   $env:CUDA_PATH   = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
#   $env:CUDNN_ROOT  = "C:\tools\cudnn-windows-x86_64-9.21.0.82_cuda12-archive"
#   $env:CC_CUDA     = "75"   # Turing; use 86 for Ampere, 89 for Ada, 90 for Hopper
#   .\build_lc0.ps1

$ErrorActionPreference = "Stop"

$RepoRoot    = Split-Path -Parent $MyInvocation.MyCommand.Path
$Lc0Dir      = Join-Path $RepoRoot "lc0-upstream"
$BuildDir    = Join-Path $Lc0Dir "build\release"
$VenvScripts = Join-Path $RepoRoot ".venv\Scripts"

# Allow override via env vars; fall back to sane defaults.
$CudaPath  = if ($env:CUDA_PATH)  { $env:CUDA_PATH }  else { "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9" }
$CudnnRoot = if ($env:CUDNN_ROOT) { $env:CUDNN_ROOT } else { "C:\tools\cudnn-windows-x86_64-9.21.0.82_cuda12-archive" }
$CcCuda    = if ($env:CC_CUDA)    { $env:CC_CUDA }    else { "86" }  # Ampere (RTX 30-series)
$VcVars    = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"

# ---- sanity checks ----
if (-not (Test-Path $VcVars))   { Write-Error "vcvarsall.bat not found at: $VcVars" }
if (-not (Test-Path "$CudaPath\bin\nvcc.exe")) { Write-Error "nvcc not found under: $CudaPath" }
if (-not (Test-Path "$CudnnRoot\include\cudnn.h")) { Write-Error "cudnn.h not found under: $CudnnRoot\include" }
if (-not (Test-Path "$Lc0Dir\meson.build")) { Write-Error "lc0 source not found at: $Lc0Dir" }

Write-Host "============================================================"
Write-Host "[1/4] Importing VS 2022 BuildTools env (amd64)..."
Write-Host "============================================================"

# Run vcvarsall in a child cmd and dump resulting env vars; re-import into this PS session.
$tmp = [System.IO.Path]::GetTempFileName()
cmd.exe /c "`"$VcVars`" amd64 && set > `"$tmp`""
if ($LASTEXITCODE -ne 0) { Write-Error "vcvarsall failed (exit $LASTEXITCODE)" }
Get-Content $tmp | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        Set-Item -Path "Env:$($matches[1])" -Value $matches[2]
    }
}
Remove-Item $tmp -Force

# Put meson.exe, ninja.exe (venv) + CUDA bin at the front of PATH.
$env:PATH = "$VenvScripts;$CudaPath\bin;$env:PATH"

# Sanity: make sure cl.exe (MSVC) is now visible.
$clExe = (Get-Command cl.exe -ErrorAction SilentlyContinue).Source
if (-not $clExe) { Write-Error "cl.exe (MSVC) not found after vcvarsall import." }
Write-Host "cl.exe: $clExe"
Write-Host "meson:  $((Get-Command meson.exe).Source)"
Write-Host "ninja:  $((Get-Command ninja.exe).Source)"
Write-Host "nvcc:   $((Get-Command nvcc.exe).Source)"

Write-Host ""
Write-Host "============================================================"
Write-Host "[2/4] Cleaning previous build (if any)..."
Write-Host "============================================================"
if (Test-Path "$Lc0Dir\build") { Remove-Item "$Lc0Dir\build" -Recurse -Force }

Write-Host ""
Write-Host "============================================================"
Write-Host "[3/4] meson setup..."
Write-Host "============================================================"
Push-Location $Lc0Dir
try {
    # Meson array options accept comma-separated values. Use forward slashes in paths.
    $cudnnInc  = "$CudaPath/include,$CudnnRoot/include" -replace '\\','/'
    $cudnnLibs = "$CudaPath/lib/x64,$CudnnRoot/lib/x64" -replace '\\','/'

    $mesonArgs = @(
        "setup", $BuildDir,
        "--backend=ninja",
        "--buildtype=release",
        "-Dcudnn=true",
        "-Dplain_cuda=true",
        "-Dcudnn_include=$cudnnInc",
        "-Dcudnn_libdirs=$cudnnLibs",
        "-Dopencl=false",
        "-Ddx=false",
        "-Dblas=false",
        "-Dopenblas=false",
        "-Dmkl=false",
        "-Ddnnl=false",
        "-Donednn=false",
        "-Dtensorflow=false",
        "-Daccelerate=false",
        "-Dispc=false",
        "-Dgtest=false",
        "-Dcc_cuda=$CcCuda",
        "-Ddefault_library=static"
    )
    & meson.exe @mesonArgs
    if ($LASTEXITCODE -ne 0) { throw "meson setup failed with code $LASTEXITCODE" }

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "[4/4] Compiling with ninja (this will take a while)..."
    Write-Host "============================================================"
    & ninja.exe -C $BuildDir
    if ($LASTEXITCODE -ne 0) { throw "ninja build failed with code $LASTEXITCODE" }
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "============================================================"
Write-Host "SUCCESS -- lc0.exe built"
Write-Host "============================================================"
$exe = Join-Path $BuildDir "lc0.exe"
if (-not (Test-Path $exe)) { Write-Error "Build reported success but $exe is missing!" }
Get-Item $exe | Select-Object FullName, Length | Format-Table -AutoSize

Write-Host ""
Write-Host "Copying cuDNN runtime DLLs next to lc0.exe..."
Copy-Item "$CudnnRoot\bin\x64\*.dll" $BuildDir -Force
Get-ChildItem "$BuildDir\cudnn*.dll" | Select-Object Name, Length | Format-Table -AutoSize
Write-Host "All done."
