# make_submission.ps1 -- produce a clean .zip for submission
#
# Strips out: .venv, build artefacts, bytecode caches, large networks,
# non-canonical models. Keeps: source code, docs, the authors'
# pretrained model, the training scaler, demo trees, labelled moves.
#
# Output: submission.zip in the repo root.

$ErrorActionPreference = "Stop"

$RepoRoot   = Split-Path -Parent $MyInvocation.MyCommand.Path
$StagingDir = Join-Path $env:TEMP "brilliant-moves-submission-$(Get-Random)"
$OutZip     = Join-Path $RepoRoot "submission.zip"

Write-Host "Staging at: $StagingDir"
New-Item -ItemType Directory -Path $StagingDir | Out-Null

# Files to include at the top level.
$topLevel = @(
    "README.md",
    "REPORT.md",
    "REPORT.tex",
    "requirements.txt",
    "build_lc0.ps1",
    "build_lc0_incremental.ps1",
    "email-to-author-DRAFT.md",
    ".gitignore",
    "make_submission.ps1"
)
foreach ($f in $topLevel) {
    $src = Join-Path $RepoRoot $f
    if (Test-Path $src) { Copy-Item $src $StagingDir }
}
# Include the PDF only if you've compiled it.
$pdf = Join-Path $RepoRoot "REPORT.pdf"
if (Test-Path $pdf) { Copy-Item $pdf $StagingDir }

# ---- lc0-upstream: source only, no build dir ----
$srcLc0 = Join-Path $RepoRoot "lc0-upstream"
$dstLc0 = Join-Path $StagingDir "lc0-upstream"
New-Item -ItemType Directory -Path $dstLc0 | Out-Null
Get-ChildItem $srcLc0 -Exclude "build","subprojects" | Copy-Item -Destination $dstLc0 -Recurse -Force
# Keep subprojects *.wrap descriptors (tiny, required by meson) but drop extracted sources.
$subSrc = Join-Path $srcLc0 "subprojects"
if (Test-Path $subSrc) {
    $subDst = Join-Path $dstLc0 "subprojects"
    New-Item -ItemType Directory -Path $subDst | Out-Null
    Get-ChildItem $subSrc -Filter "*.wrap" | Copy-Item -Destination $subDst -Force
}

# ---- brilliant-moves-clf: source + demo data, no lc0 build, no weights, no throwaway training artefacts ----
$srcClf = Join-Path $RepoRoot "brilliant-moves-clf"
$dstClf = Join-Path $StagingDir "brilliant-moves-clf"
Copy-Item $srcClf $dstClf -Recurse -Force
# Prune what grader should regenerate or download separately.
Remove-Item -Recurse -Force (Join-Path $dstClf "brilliant_moves_clf\lc0\build") -EA SilentlyContinue
Remove-Item -Recurse -Force (Join-Path $dstClf "brilliant_moves_clf\__pycache__") -EA SilentlyContinue
Get-ChildItem (Join-Path $dstClf "brilliant_moves_clf\models") -Filter "model_*.pth" -EA SilentlyContinue | Remove-Item -Force
Get-ChildItem (Join-Path $dstClf "brilliant_moves_clf\models") -Filter "scaler_*.pkl" -EA SilentlyContinue | Remove-Item -Force
Get-ChildItem (Join-Path $dstClf "brilliant_moves_clf\models") -Filter "train_log_*.json" -EA SilentlyContinue | Remove-Item -Force
# Keep: models\model-7936-2.pth (pretrained) and models\scaler.pkl (our fix artefact).

# Weights are large network files -- grader re-downloads via instructions in README.
$weightsDir = Join-Path $dstClf "brilliant_moves_clf\weights"
if (Test-Path $weightsDir) {
    Get-ChildItem $weightsDir -Include "*.pb","*.pb.gz" -Recurse | Remove-Item -Force
    # leave a placeholder so the directory survives zip
    if (-not (Get-ChildItem $weightsDir)) {
        Set-Content -Path (Join-Path $weightsDir "README.txt") -Value "Download Leela T82 and Maia networks; see top-level README."
    }
}

Write-Host ""
Write-Host "Staged contents:"
Get-ChildItem $StagingDir | Select-Object Name, @{N='Size_MB';E={[math]::Round((Get-ChildItem $_.FullName -Recurse -File -EA SilentlyContinue | Measure-Object Length -Sum).Sum / 1MB, 1)}} | Format-Table -AutoSize

# Compress
if (Test-Path $OutZip) { Remove-Item $OutZip -Force }
Compress-Archive -Path "$StagingDir\*" -DestinationPath $OutZip -CompressionLevel Optimal
$size = (Get-Item $OutZip).Length / 1MB
Write-Host ""
Write-Host "Wrote: $OutZip ($([math]::Round($size,1)) MB)"

Remove-Item -Recurse -Force $StagingDir
Write-Host "Done."
