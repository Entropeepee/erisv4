# keep_eris_online.ps1
# ---------------------------------------------------------------------------
# Keeps Eris online without hogging the machine when you're busy.
# Run it on a schedule (see KEEP_ERIS_ONLINE.md — e.g. 3:00 AM and 3:00 PM).
#
# What it does:
#   1. If Eris already answers on http://localhost:8001 -> do nothing.
#   2. Pre-flight safety:
#        - LOW DISK: if free space < -MinFreeDiskGB, refuse to start (avoids
#          filling the drive / crashing the system).
#        - BUSY GPU: if VRAM in use > -VramBusyGB (you're likely gaming), the
#          countdown DEFAULTS TO CANCEL so it won't steal the GPU unless you say so.
#   3. Shows a cancelable countdown popup (default -CountdownSeconds).
#   4. If not cancelled: launches start_eris.bat (which starts Ollama + the
#      backend + the cockpit) and waits until Eris answers.
# Nothing is forced: every risky case shows you the popup first.
# ---------------------------------------------------------------------------
param(
  [int]$CountdownSeconds = 30,
  [double]$MinFreeDiskGB = 1.5,    # refuse to start below this free disk
  [double]$VramBusyGB    = 3.0,    # above this VRAM-in-use, default to CANCEL
  [string]$ErisDir       = "C:\Users\david\.gemini\antigravity\scratch\erisv4",
  [string]$HealthUrl     = "http://localhost:8001/api/system"
)

function Test-ErisOnline {
  try {
    $r = Invoke-WebRequest -Uri $HealthUrl -TimeoutSec 4 -UseBasicParsing -ErrorAction Stop
    return ($r.StatusCode -eq 200)
  } catch { return $false }
}

function Get-FreeDiskGB([string]$path) {
  try { return [math]::Round((Get-PSDrive (Get-Item $path).PSDrive.Name).Free / 1GB, 2) }
  catch { return 9999 }   # unknown -> don't block
}

function Get-VramUsedGB {
  try {
    $u = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
    if ($u) { return [math]::Round([double]$u / 1024.0, 2) }
  } catch { }
  return 0
}

function Show-Countdown([string]$msg, [int]$secs, [bool]$defaultCancel) {
  Add-Type -AssemblyName System.Windows.Forms
  Add-Type -AssemblyName System.Drawing
  $form = New-Object System.Windows.Forms.Form
  $form.Text = "Eris auto-start"; $form.TopMost = $true
  $form.Size = New-Object System.Drawing.Size(460, 210)
  $form.StartPosition = "CenterScreen"; $form.FormBorderStyle = "FixedDialog"
  $label = New-Object System.Windows.Forms.Label
  $label.Size = New-Object System.Drawing.Size(430, 100)
  $label.Location = New-Object System.Drawing.Point(15, 12)
  $form.Controls.Add($label)
  $start = New-Object System.Windows.Forms.Button
  $start.Text = "Start Eris now"; $start.Size = New-Object System.Drawing.Size(140, 32)
  $start.Location = New-Object System.Drawing.Point(70, 125)
  $start.DialogResult = [System.Windows.Forms.DialogResult]::Yes
  $form.Controls.Add($start)
  $cancel = New-Object System.Windows.Forms.Button
  $cancel.Text = "Cancel"; $cancel.Size = New-Object System.Drawing.Size(140, 32)
  $cancel.Location = New-Object System.Drawing.Point(240, 125)
  $cancel.DialogResult = [System.Windows.Forms.DialogResult]::No
  $form.Controls.Add($cancel)
  $form.AcceptButton = if ($defaultCancel) { $cancel } else { $start }
  $verb = if ($defaultCancel) { "CANCEL" } else { "start" }
  $script:left = $secs
  $label.Text = "$msg`r`n`r`nAuto-$verb in $secs seconds..."
  $timer = New-Object System.Windows.Forms.Timer
  $timer.Interval = 1000
  $timer.Add_Tick({
    $script:left--
    $label.Text = "$msg`r`n`r`nAuto-$verb in $script:left seconds..."
    if ($script:left -le 0) {
      $timer.Stop()
      $form.DialogResult = if ($defaultCancel) { [System.Windows.Forms.DialogResult]::No } else { [System.Windows.Forms.DialogResult]::Yes }
      $form.Close()
    }
  })
  $timer.Start()
  $res = $form.ShowDialog()
  $timer.Stop()
  return ($res -eq [System.Windows.Forms.DialogResult]::Yes)
}

# ── 1. Already online? ──────────────────────────────────────────────────────
if (Test-ErisOnline) { Write-Host "Eris is already online."; exit 0 }

# ── 2a. Low disk -> refuse ──────────────────────────────────────────────────
$freeGB = Get-FreeDiskGB $ErisDir
if ($freeGB -lt $MinFreeDiskGB) {
  Add-Type -AssemblyName System.Windows.Forms
  [System.Windows.Forms.MessageBox]::Show(
    "Not starting Eris: only $freeGB GB free (need $MinFreeDiskGB GB). Free up disk space first.",
    "Eris paused - low disk") | Out-Null
  exit 1
}

# ── 2b. Busy GPU -> default to cancel ───────────────────────────────────────
$vramGB = Get-VramUsedGB
$gpuBusy = $vramGB -gt $VramBusyGB
$msg = "Eris is offline. Bring her online (Ollama + backend + cockpit)?`r`n" +
       "Free disk: $freeGB GB   VRAM in use: $vramGB GB"
if ($gpuBusy) {
  $msg += "`r`n`r`nNOTE: the GPU looks busy (likely a game) - this will CANCEL unless you click Start."
}

# ── 3. Countdown popup ──────────────────────────────────────────────────────
if (-not (Show-Countdown $msg $CountdownSeconds $gpuBusy)) {
  Write-Host "Startup cancelled by user / GPU busy."
  exit 0
}

# ── 4. Launch + wait for ready ──────────────────────────────────────────────
Write-Host "Starting Eris..."
Start-Process -FilePath (Join-Path $ErisDir "start_eris.bat") -WorkingDirectory $ErisDir
for ($i = 0; $i -lt 60; $i++) {
  Start-Sleep -Seconds 3
  if (Test-ErisOnline) { Write-Host "Eris is online and ready."; exit 0 }
}
Write-Host "Launched start_eris.bat, but Eris did not answer within ~3 min. Check the backend window."
exit 1
