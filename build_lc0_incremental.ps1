# Incremental ninja rebuild (reuses existing meson setup)
$ErrorActionPreference = "Stop"

$RepoRoot    = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildDir    = Join-Path $RepoRoot "lc0-upstream\build\release"
$VenvScripts = Join-Path $RepoRoot ".venv\Scripts"
$CudaPath    = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
$CudnnRoot   = "C:\Users\joash\Downloads\cudnn-windows-x86_64-9.21.0.82_cuda12-archive\cudnn-windows-x86_64-9.21.0.82_cuda12-archive"
$VcVars      = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"

Write-Host "[1/2] Importing VS 2022 BuildTools env (amd64)..."
$tmp = [System.IO.Path]::GetTempFileName()
cmd.exe /c "`"$VcVars`" amd64 && set > `"$tmp`""
if ($LASTEXITCODE -ne 0) { Write-Error "vcvarsall failed ($LASTEXITCODE)" }
Get-Content $tmp | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "Env:$($matches[1])" -Value $matches[2] }
}
Remove-Item $tmp -Force
$env:PATH = "$VenvScripts;$CudaPath\bin;$env:PATH"

Write-Host "[2/2] ninja (incremental)..."
& ninja.exe -C $BuildDir
if ($LASTEXITCODE -ne 0) { Write-Error "ninja failed ($LASTEXITCODE)" }

$exe = Join-Path $BuildDir "lc0.exe"
if (-not (Test-Path $exe)) { Write-Error "$exe missing!" }
Write-Host "`nBuilt lc0.exe:"
Get-Item $exe | Select-Object FullName, Length | Format-Table -AutoSize

Write-Host "Copying cuDNN runtime DLLs next to lc0.exe..."
Copy-Item "$CudnnRoot\bin\x64\*.dll" $BuildDir -Force
Write-Host "Done."
