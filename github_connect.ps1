param(
    [Parameter(Mandatory = $true)]
    [string]$RemoteUrl,

    [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path ".git")) {
    git init
}

if (-not (git config user.name)) {
    Write-Host "Git user.name is not set. Run: git config --global user.name \"Your Name\""
    exit 1
}

if (-not (git config user.email)) {
    Write-Host "Git user.email is not set. Run: git config --global user.email \"you@example.com\""
    exit 1
}

git add .

$hasCommit = $true
try {
    git rev-parse --verify HEAD *> $null
} catch {
    $hasCommit = $false
}

if ($hasCommit) {
    git commit -m "Update project for cloud and HD pipeline" --allow-empty
} else {
    git commit -m "Initial commit"
}

$originExists = $false
try {
    $originUrl = git remote get-url origin
    if ($originUrl) {
        $originExists = $true
    }
} catch {
    $originExists = $false
}

if ($originExists) {
    git remote set-url origin $RemoteUrl
} else {
    git remote add origin $RemoteUrl
}

git branch -M $Branch
git push -u origin $Branch

Write-Host "GitHub remote connected and pushed to $Branch"
