# Keep Eris online (scheduled, with safeguards)

`keep_eris_online.ps1` brings Eris up if she's down, but **only when it's safe**,
and always lets you cancel. Schedule it to run a couple of times a day so she
stays available, learning and dreaming, without surprising you mid-game.

## What it protects against
- **Already running** → it does nothing (won't double-launch).
- **Low disk** → if free space on her drive is under `-MinFreeDiskGB` (default
  1.5 GB) it refuses to start and tells you, so a full crawl can't fill the disk
  and destabilize the system.
- **Busy GPU** → if more than `-VramBusyGB` (default 3 GB) of VRAM is already in
  use (you're probably gaming), the countdown **defaults to CANCEL** — it won't
  take the GPU unless you click *Start*.
- **Always cancelable** → a popup with a `-CountdownSeconds` (default 30s)
  countdown. If you're busy, ignore it (or click Cancel) and it backs off.

## Schedule it (twice daily — 3:00 AM and 3:00 PM)
Open **PowerShell** and paste this once (creates the scheduled task; runs in your
session so the popup can appear):

```powershell
$dir = "C:\Users\david\.gemini\antigravity\scratch\erisv4"
$action  = New-ScheduledTaskAction -Execute "powershell.exe" `
  -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$dir\keep_eris_online.ps1`""
$t1 = New-ScheduledTaskTrigger -Daily -At 3:00AM
$t2 = New-ScheduledTaskTrigger -Daily -At 3:00PM
$set = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
Register-ScheduledTask -TaskName "Keep Eris Online" -Action $action `
  -Trigger $t1,$t2 -Settings $set -Description "Bring Eris online if down (with safeguards)."
```

To **test it now** without waiting:
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\david\.gemini\antigravity\scratch\erisv4\keep_eris_online.ps1"
```

To **change times / thresholds**, re-run with different values, e.g.:
```powershell
... -File "...\keep_eris_online.ps1" -CountdownSeconds 45 -MinFreeDiskGB 3 -VramBusyGB 2
```

To **remove the schedule**:
```powershell
Unregister-ScheduledTask -TaskName "Keep Eris Online" -Confirm:$false
```

## Stopping the crawling / studying (the "don't run wild" switches)
If you ever want her online but **not** crawling/researching/writing in the
background, launch with these environment variables (set them before
`start_eris.bat`, or add `set NAME=VALUE` lines near the top of it):

| Switch | Effect |
|---|---|
| `ERIS_IDLE_READING=0` | Stops the self-directed web crawling / learning loop. |
| `ERIS_STUDY_ENABLED=0` | Stops the nightly study session. |
| `ERIS_CRAWL_PERIOD_S=3600` | Slows the background loop (here: ~1×/hour). |
| `ERIS_GPU=0` | Keeps her field math on CPU (hand the GPU to a game / Unreal). |
| `ERIS_LOCAL_MODEL=mistral` | Uses the smaller ~4.4 GB model (frees VRAM). |

She also **guards the disk herself**: document authoring and ingestion skip
writing when free space drops below `ERIS_MIN_DISK_GB` (default 1 GB), so she
won't fill the last of your drive even if a session runs long.
