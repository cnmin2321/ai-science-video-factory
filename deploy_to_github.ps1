# 在已登录 GitHub CLI 的机器上，从本目录创建私有仓库并推送
# 用法：powershell -ExecutionPolicy Bypass -File .\deploy_to_github.ps1 -RepoName ai-science-video-factory
param(
    [string]$RepoName = "ai-science-video-factory",
    [switch]$Public
)

$ErrorActionPreference = "Stop"
# 项目内便携版优先，避免依赖系统 PATH
$projectRoot = Split-Path -Parent $PSScriptRoot
$gitDir = Join-Path $projectRoot "tools\git\cmd"
$ghDir = Join-Path $projectRoot "tools\gh\bin"
if (Test-Path (Join-Path $gitDir "git.exe")) { $env:Path = "$gitDir;$env:Path" }
if (Test-Path (Join-Path $ghDir "gh.exe")) { $env:Path = "$ghDir;$env:Path" }
$env:GH_CONFIG_DIR = Join-Path $projectRoot "runtime\gh-config"
New-Item -ItemType Directory -Force -Path $env:GH_CONFIG_DIR | Out-Null
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "未找到 git。请把本目录文件上传到 GitHub 仓库，或在项目 tools 目录安装 Git。"
}
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "未找到 GitHub CLI (gh)。请先在项目 tools 目录安装 gh。"
}
gh auth status
if (-not (Test-Path ".git")) {
    git init
    git branch -M main
}
git add .
git commit -m "feat: GitHub-ready single-file AI science video factory" 2>$null
if ($LASTEXITCODE -ne 0) { Write-Host "没有新的改动需要提交，继续推送。" }
$visibility = if ($Public) { "--public" } else { "--private" }
gh repo create $RepoName $visibility --source . --remote origin --push
Write-Host "完成：https://github.com/$((gh api user --jq .login))/$RepoName"
