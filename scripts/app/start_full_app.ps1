param(
    [string]$BindHost = "127.0.0.1",
    [int]$BindPort = 8000,
    [switch]$Reload,
    [ValidateSet("yolov8", "yolo_world")]
    [string]$DetectorBackend = "yolov8",
    [string]$ModelWeights = "",
    [string]$RefinerModel = "",
    [string]$RouterModel = "",
    [switch]$SkipOllama,
    [switch]$SkipVectorCheck,
    [switch]$AutoPullModel,
    [switch]$SkipXmpp
)

$argsList = @(
    "scripts/app/start_full_app.py",
    "--host", $BindHost,
    "--port", $BindPort,
    "--detector-backend", $DetectorBackend
)

if ($Reload) {
    $argsList += "--reload"
}

if ($ModelWeights) {
    $argsList += @("--model-weights", $ModelWeights)
}

if ($RefinerModel) {
    $argsList += @("--refiner-model", $RefinerModel)
}

if ($RouterModel) {
    $argsList += @("--router-model", $RouterModel)
}

if ($SkipOllama) {
    $argsList += "--skip-ollama"
}

if ($SkipVectorCheck) {
    $argsList += "--skip-vector-check"
}

if ($AutoPullModel) {
    $argsList += "--auto-pull-model"
}

if ($SkipXmpp) {
    $argsList += "--skip-xmpp"
}

python @argsList
exit $LASTEXITCODE
