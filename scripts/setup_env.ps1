# LiteAVSE 环境配置脚本 (Windows PowerShell)
# 在项目根目录执行: .\scripts\setup_env.ps1

$ErrorActionPreference = "Stop"
# 脚本在 scripts/ 下，项目根目录为上一级
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$venvPath = Join-Path $ProjectRoot ".venv"
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"

# 1. 创建虚拟环境（若不存在）
if (-not (Test-Path $activateScript)) {
    Write-Host "正在创建虚拟环境 .venv ..." -ForegroundColor Cyan
    python -m venv $venvPath
    if ($LASTEXITCODE -ne 0) { throw "创建 venv 失败" }
}

# 2. 激活并升级 pip
Write-Host "激活虚拟环境并升级 pip ..." -ForegroundColor Cyan
& $activateScript
python -m pip install --upgrade pip --quiet

# 3. 安装核心依赖（不含 pesq，避免未装 C++ 编译环境时报错）
$coreReq = Join-Path $ProjectRoot "requirements_core.txt"
if (Test-Path $coreReq) {
    Write-Host "正在安装 requirements_core.txt ..." -ForegroundColor Cyan
    pip install -r $coreReq
    if ($LASTEXITCODE -ne 0) { throw "安装 requirements_core.txt 失败" }
} else {
    Write-Host "未找到 requirements_core.txt，尝试安装 requirements.txt（可能因 pesq 编译失败）..." -ForegroundColor Yellow
    pip install -r (Join-Path $ProjectRoot "requirements.txt")
}

Write-Host ""
Write-Host "核心依赖安装完成。" -ForegroundColor Green
Write-Host "请阅读 SETUP_CN.md 完成：" -ForegroundColor Yellow
Write-Host "  1. 安装 PESQ（需先安装 Microsoft C++ Build Tools）" -ForegroundColor Yellow
Write-Host "  2. 安装 mamba-ssm（Windows 建议从源码或使用 WSL）" -ForegroundColor Yellow
