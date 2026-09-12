<#
.SYNOPSIS
Follow the running A/B match without having to find its output file.

.DESCRIPTION
Match output lands in a log file whose name changes every run, so this picks
the most recently written one under -Root and follows it. Ctrl+C stops
watching and does not touch the match.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File tools\watch_match.ps1
    powershell -ExecutionPolicy Bypass -File tools\watch_match.ps1 -Summary
    powershell -ExecutionPolicy Bypass -File tools\watch_match.ps1 -Tail 40
#>
param(
    # Where the harness writes its logs.
    [string]$Root = "$env:TEMP",
    # Lines of history to show before following.
    [int]$Tail = 15,
    # Show only the running score and LLR, one line per update, instead of
    # every game. Useful when the match is long and the per-game lines scroll.
    [switch]$Summary
)

$log = Get-ChildItem -Path $Root -Filter *.output -Recurse -ErrorAction SilentlyContinue |
       Where-Object { $_.Length -gt 0 } |
       Sort-Object LastWriteTime -Descending |
       Select-Object -First 1

if (-not $log) {
    Write-Host "no match log found under $Root" -ForegroundColor Yellow
    exit 1
}

Write-Host ("following {0}" -f $log.FullName) -ForegroundColor Cyan
Write-Host ("last written {0}" -f $log.LastWriteTime) -ForegroundColor DarkGray
Write-Host ""

if ($Summary) {
    # One line per 25 games, plus the verdict lines at the end.
    Get-Content $log.FullName -Tail $Tail -Wait | ForEach-Object {
        if ($_ -match "game (\d+)/" -and [int]$Matches[1] % 25 -eq 0) { $_ }
        elseif ($_ -match "score |LLR |elo |SPRT ") { $_ }
    }
} else {
    Get-Content $log.FullName -Tail $Tail -Wait
}
