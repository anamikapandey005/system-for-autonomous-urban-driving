Write-Host "1. Downloading Python 3.8.10..."
Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.8.10/python-3.8.10-amd64.exe" -OutFile "python-3.8.10-amd64.exe"

Write-Host "2. Installing Python 3.8.10 silently..."
Start-Process -FilePath ".\python-3.8.10-amd64.exe" -ArgumentList "/quiet InstallAllUsers=0 PrependPath=1 Include_test=0" -Wait

$py38Path = "$env:USERPROFILE\AppData\Local\Programs\Python\Python38\python.exe"

Write-Host "3. Creating Virtual Environment with Python 3.8..."
& $py38Path -m venv venv

Write-Host "4. Installing dependencies in venv..."
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "5. Starting Background Download for CARLA 0.9.15 (5GB). This will take a while!"
Start-Job -ScriptBlock {
    Invoke-WebRequest -Uri "https://carla-releases.s3.us-east-005.backblazeb2.com/Windows/CARLA_0.9.15.zip" -OutFile "d:\Tesla\CARLA_0.9.15.zip"
}
Write-Host "Setup script finished successfully. CARLA download is running in the background."
