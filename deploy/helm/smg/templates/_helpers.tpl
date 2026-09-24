{{/*
Expand the name of the chart.
*/}}
{{- define "smg.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "smg.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 45 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 45 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 45 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "smg.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "smg.labels" -}}
helm.sh/chart: {{ include "smg.chart" . }}
{{ include "smg.selectorLabels" . }}
app.kubernetes.io/version: {{ .Values.global.image.tag | default .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "smg.selectorLabels" -}}
app.kubernetes.io/name: {{ include "smg.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Router image
*/}}
{{- define "smg.routerImage" -}}
{{- $registry := .Values.global.image.registry -}}
{{- $repository := .Values.global.image.repository -}}
{{- $tag := .Values.global.image.tag | default .Chart.AppVersion -}}
{{- printf "%s/%s:%s" $registry $repository $tag -}}
{{- end }}

{{/*
Service account name
*/}}
{{- define "smg.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "smg.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Router CLI arguments -- builds the full args list from values.
Called from the router Deployment template.
*/}}
{{- define "smg.routerArgs" -}}
- "--host"
- "0.0.0.0"
- "--port"
- {{ .Values.router.port | quote }}
- "--policy"
- {{ .Values.router.policy | quote }}
{{- if .Values.router.enableIgw }}
- "--enable-igw"
{{- end }}
{{- if .Values.router.mesh.enabled }}
- "--enable-mesh"
- "--mesh-host"
- "$(POD_IP)"
- "--mesh-port"
- {{ .Values.router.mesh.port | quote }}
- "--mesh-server-name"
- "$(POD_NAME)"
{{- if .Values.router.mesh.peerUrls }}
{{- range .Values.router.mesh.peerUrls }}
- "--mesh-peer-urls"
- {{ . | quote }}
{{- end }}
{{- else }}
{{/* Use K8s service discovery to find other router pods by label.
     Each router pod is annotated with the mesh port so peers can connect.
     This avoids the SocketAddr limitation that prevents DNS hostnames
     in --mesh-peer-urls. */}}
- "--service-discovery"
- "--selector"
- "app.kubernetes.io/instance={{ .Release.Name }}"
- "app.kubernetes.io/component=worker"
- "--router-selector"
- "app.kubernetes.io/component=router"
- "--service-discovery-namespace"
- {{ .Release.Namespace | quote }}
{{- if .Values.workers }}
{{- $firstWorker := index .Values.workers 0 }}
{{- $workerDefaults := index .Values.engineDefaults $firstWorker.engine }}
- "--service-discovery-port"
- {{ $firstWorker.port | default $workerDefaults.port | quote }}
{{- end }}
{{- end }}
{{- end }}
{{- if .Values.router.workerUrls }}
- "--worker-urls"
{{- range .Values.router.workerUrls }}
- {{ . | quote }}
{{- end }}
{{- else if and .Values.workers (not .Values.router.serviceDiscovery.enabled) (not (and .Values.router.mesh.enabled (not .Values.router.mesh.peerUrls))) }}
- "--worker-urls"
{{- range $i, $worker := .Values.workers }}
{{- $defaults := index $.Values.engineDefaults $worker.engine }}
{{- $port := $worker.port | default $defaults.port }}
{{- $mode := $worker.connectionMode | default "http" }}
{{- $scheme := "http" }}
{{- if eq $mode "grpc" }}
{{- $scheme = "grpc" }}
{{- end }}
- "{{ $scheme }}://{{ include "smg.fullname" $ }}-worker-{{ $worker.name }}:{{ $port }}"
{{- end }}
{{- end }}
{{- if .Values.router.serviceDiscovery.enabled }}
- "--service-discovery"
{{- if .Values.router.serviceDiscovery.selector }}
- "--selector"
- {{ .Values.router.serviceDiscovery.selector | quote }}
{{- end }}
- "--service-discovery-port"
- {{ .Values.router.serviceDiscovery.port | quote }}
{{- if not .Values.router.serviceDiscovery.clusterWide }}
- "--service-discovery-namespace"
- {{ .Values.router.serviceDiscovery.namespace | default .Release.Namespace | quote }}
{{- end }}
{{- if .Values.router.serviceDiscovery.modelIdFrom }}
- "--model-id-from"
- {{ .Values.router.serviceDiscovery.modelIdFrom | quote }}
{{- end }}
{{- end }}
{{- if .Values.router.model }}
- "--model-path"
- {{ .Values.router.model | quote }}
{{- end }}
{{- if .Values.router.tokenizerPath }}
- "--tokenizer-path"
- {{ .Values.router.tokenizerPath | quote }}
{{- end }}
{{- if .Values.router.chatTemplate }}
- "--chat-template"
- {{ .Values.router.chatTemplate | quote }}
{{- end }}
- "--cache-threshold"
- {{ .Values.router.cacheThreshold | quote }}
- "--balance-abs-threshold"
- {{ .Values.router.balanceAbsThreshold | quote }}
- "--balance-rel-threshold"
- {{ .Values.router.balanceRelThreshold | quote }}
- "--eviction-interval-secs"
- {{ .Values.router.evictionIntervalSecs | quote }}
- "--max-tree-size"
- {{ int .Values.router.maxTreeSize | quote }}
- "--block-size"
- {{ .Values.router.blockSize | quote }}
- "--max-payload-size"
- {{ int .Values.router.maxPayloadSize | quote }}
- "--request-timeout-secs"
- {{ .Values.router.requestTimeoutSecs | quote }}
- "--max-concurrent-requests"
- {{ .Values.router.maxConcurrentRequests | quote }}
- "--queue-size"
- {{ .Values.router.queueSize | quote }}
- "--queue-timeout-secs"
- {{ .Values.router.queueTimeoutSecs | quote }}
{{- if not .Values.router.retry.enabled }}
- "--disable-retries"
{{- else }}
- "--retry-max-retries"
- {{ .Values.router.retry.maxRetries | quote }}
- "--retry-initial-backoff-ms"
- {{ .Values.router.retry.initialBackoffMs | quote }}
- "--retry-max-backoff-ms"
- {{ .Values.router.retry.maxBackoffMs | quote }}
- "--retry-backoff-multiplier"
- {{ .Values.router.retry.backoffMultiplier | quote }}
- "--retry-jitter-factor"
- {{ .Values.router.retry.jitterFactor | quote }}
{{- end }}
{{- if not .Values.router.circuitBreaker.enabled }}
- "--disable-circuit-breaker"
{{- else }}
- "--cb-failure-threshold"
- {{ .Values.router.circuitBreaker.failureThreshold | quote }}
- "--cb-success-threshold"
- {{ .Values.router.circuitBreaker.successThreshold | quote }}
- "--cb-timeout-duration-secs"
- {{ .Values.router.circuitBreaker.timeoutDurationSecs | quote }}
- "--cb-window-duration-secs"
- {{ .Values.router.circuitBreaker.windowDurationSecs | quote }}
{{- end }}
{{- if not .Values.router.healthCheck.enabled }}
- "--disable-health-check"
{{- else }}
- "--health-failure-threshold"
- {{ .Values.router.healthCheck.failureThreshold | quote }}
- "--health-success-threshold"
- {{ .Values.router.healthCheck.successThreshold | quote }}
- "--health-check-timeout-secs"
- {{ .Values.router.healthCheck.timeoutSecs | quote }}
- "--health-check-interval-secs"
- {{ .Values.router.healthCheck.intervalSecs | quote }}
- "--health-check-endpoint"
- {{ .Values.router.healthCheck.endpoint | quote }}
{{- end }}
- "--prometheus-port"
- {{ .Values.router.metrics.port | quote }}
- "--log-level"
- {{ .Values.router.logging.level | quote }}
{{- if .Values.router.logging.json }}
- "--log-json"
{{- end }}
{{- if .Values.router.logging.dir }}
- "--log-dir"
- {{ .Values.router.logging.dir | quote }}
{{- end }}
{{- if .Values.router.tracing.enabled }}
- "--enable-trace"
- "--otlp-traces-endpoint"
- {{ .Values.router.tracing.otlpEndpoint | quote }}
{{- end }}
{{- if ne .Values.history.backend "memory" }}
- "--history-backend"
- {{ .Values.history.backend | quote }}
{{- end }}
{{- if eq .Values.history.backend "postgres" }}
{{- if .Values.history.postgres.url }}
- "--postgres-db-url"
- {{ .Values.history.postgres.url | quote }}
{{- end }}
- "--postgres-pool-max"
- {{ .Values.history.postgres.poolMax | quote }}
{{- end }}
{{- if eq .Values.history.backend "redis" }}
{{- if .Values.history.redis.url }}
- "--redis-url"
- {{ .Values.history.redis.url | quote }}
{{- end }}
- "--redis-pool-max"
- {{ .Values.history.redis.poolMax | quote }}
{{- end }}
{{- if eq .Values.history.backend "oracle" }}
{{- if .Values.history.oracle.dsn }}
- "--oracle-connect-descriptor"
- {{ .Values.history.oracle.dsn | quote }}
{{- end }}
- "--oracle-pool-max"
- {{ .Values.history.oracle.poolMax | quote }}
{{- if .Values.history.oracle.user }}
- "--oracle-username"
- {{ .Values.history.oracle.user | quote }}
{{- end }}
{{- if .Values.history.oracle.password }}
- "--oracle-password"
- {{ .Values.history.oracle.password | quote }}
{{- end }}
{{- end }}
{{- if ge (int .Values.auth.rateLimitTokensPerSecond) 0 }}
- "--rate-limit-tokens-per-second"
- {{ .Values.auth.rateLimitTokensPerSecond | quote }}
{{- end }}
{{- if .Values.router.wasm.enabled }}
- "--enable-wasm"
{{- end }}
{{- if .Values.router.wasm.path }}
- "--storage-hook-wasm-path"
- {{ .Values.router.wasm.path | quote }}
{{- end }}
{{- if .Values.router.mcp.enabled }}
- "--mcp-config-path"
- {{ .Values.router.mcp.configPath | quote }}
{{- end }}
{{- if .Values.router.reasoningParser }}
- "--reasoning-parser"
- {{ .Values.router.reasoningParser | quote }}
{{- end }}
{{- if .Values.router.toolCallParser }}
- "--tool-call-parser"
- {{ .Values.router.toolCallParser | quote }}
{{- end }}
{{- if .Values.auth.apiKey }}
- "--api-key"
- {{ .Values.auth.apiKey | quote }}
{{- end }}
{{- range .Values.router.extraArgs }}
- {{ . | quote }}
{{- end }}
{{- end }}

{{/*
Worker image -- resolves the full image reference for a worker.
Expects a dict with keys: worker, global, chart, defaults
*/}}
{{- define "smg.workerImage" -}}
{{- $workerRegistry := "" -}}
{{- $workerRepository := "" -}}
{{- $workerTag := "" -}}
{{- if .worker.image -}}
{{- $workerRegistry = .worker.image.registry | default "" -}}
{{- $workerRepository = .worker.image.repository | default "" -}}
{{- $workerTag = .worker.image.tag | default "" -}}
{{- end -}}
{{/* Engine images default to ghcr.io (gateway image is on docker.io) */}}
{{- $registry := $workerRegistry | default "ghcr.io" -}}
{{- $repository := $workerRepository | default .global.image.repository -}}
{{- $tag := $workerTag | default (.global.image.tag | default .chart.AppVersion) -}}
{{- printf "%s/%s:%s" $registry $repository $tag -}}
{{- end }}

{{/*
Worker command -- returns the command array based on engine + connectionMode.
Expects a dict with keys: worker
*/}}
{{- define "smg.workerCommand" -}}
{{- $engine := .worker.engine -}}
{{- $mode := .worker.connectionMode | default "http" -}}
{{- if eq $engine "vllm" -}}
{{- if eq $mode "grpc" -}}
["python3", "-m", "vllm.entrypoints.grpc_server"]
{{- else -}}
["python3", "-m", "vllm.entrypoints.openai.api_server"]
{{- end -}}
{{- else if eq $engine "sglang" -}}
["python3", "-m", "sglang.launch_server"]
{{- end -}}
{{- end }}

{{/*
Worker args -- returns the args list based on engine.
Expects a dict with keys: worker, defaults
*/}}
{{- define "smg.workerArgs" -}}
{{- $worker := .worker -}}
{{- $defaults := index .defaults $worker.engine -}}
{{- $port := $worker.port | default $defaults.port -}}
{{- $gpuCount := 1 -}}
{{- if $worker.gpu -}}
{{- $gpuCount = $worker.gpu.count | default 1 -}}
{{- end -}}
{{- $mode := $worker.connectionMode | default "http" }}
{{- if eq $worker.engine "vllm" }}
- "--model"
- {{ $worker.model | quote }}
- "--host"
- "0.0.0.0"
- "--port"
- {{ $port | quote }}
- "--tensor-parallel-size"
- {{ $gpuCount | quote }}
{{- else if eq $worker.engine "sglang" }}
- "--model-path"
- {{ $worker.model | quote }}
- "--host"
- "0.0.0.0"
- "--port"
- {{ $port | quote }}
- "--tp-size"
- {{ $gpuCount | quote }}
{{- if eq $mode "grpc" }}
- "--grpc-mode"
{{- end }}
{{- end }}
{{- if $worker.extraArgs }}
{{- range $worker.extraArgs }}
- {{ . | quote }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Auto-generate --worker-urls from worker Services.
Used by smg.routerArgs when workers are defined but workerUrls is empty.
*/}}
{{- define "smg.autoWorkerUrls" -}}
{{- range $i, $worker := .Values.workers }}
{{- $defaults := index $.Values.engineDefaults $worker.engine }}
{{- $port := $worker.port | default $defaults.port }}
{{- $mode := $worker.connectionMode | default "http" }}
{{- $scheme := "http" }}
{{- if eq $mode "grpc" }}
{{- $scheme = "grpc" }}
{{- end }}
- "{{ $scheme }}://{{ include "smg.fullname" $ }}-worker-{{ $worker.name }}:{{ $port }}"
{{- end }}
{{- end }}
