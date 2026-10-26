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
Discovery provider -- the worker discovery provider the router runs:
router.discovery.provider, or "kubernetes" for router.serviceDiscovery.enabled,
its legacy spelling. Empty when workers are not discovered (mesh router
discovery is not worker discovery; see smg.meshRouterDiscovery).
*/}}
{{- define "smg.discoveryProvider" -}}
{{- if .Values.router.serviceDiscovery.enabled -}}
kubernetes
{{- else -}}
{{- .Values.router.discovery.provider | default "" -}}
{{- end -}}
{{- end }}

{{/*
Mesh router discovery -- "true" when mesh is on without peerUrls, so the
routers find each other among the release's router pods through the
Kubernetes API. The gateway runs that only inside Kubernetes discovery.
*/}}
{{- define "smg.meshRouterDiscovery" -}}
{{- if and .Values.router.mesh.enabled (not .Values.router.mesh.peerUrls) -}}true{{- end -}}
{{- end }}

{{/*
Kubernetes API -- "true" when the router reads pods through the Kubernetes
API: Kubernetes worker discovery, by either spelling, or mesh router
discovery. The Role and RoleBinding are created for it; the file, Slurm and
Consul providers never call the API, so they get none unless rbac.create.
*/}}
{{- define "smg.usesKubernetesApi" -}}
{{- if or (eq (include "smg.discoveryProvider" .) "kubernetes") (include "smg.meshRouterDiscovery" .) -}}true{{- end -}}
{{- end }}

{{/*
Discovery cluster-wide -- "true" when Kubernetes discovery watches every
namespace, so its RBAC is a ClusterRole and ClusterRoleBinding.
router.serviceDiscovery.clusterWide counts on its own, as it always has.
*/}}
{{- define "smg.discoveryClusterWide" -}}
{{- $discovery := .Values.router.discovery -}}
{{- if or .Values.router.serviceDiscovery.clusterWide (and (eq ($discovery.provider | default "") "kubernetes") $discovery.kubernetes.clusterWide) -}}true{{- end -}}
{{- end }}

{{/*
Discovery namespace -- the namespace Kubernetes discovery watches when it is
not cluster-wide: router.discovery.kubernetes.namespace under the provider,
router.serviceDiscovery.namespace under the legacy spelling, the release's
otherwise (mesh router discovery alone watches the release's). The Role and
RoleBinding are created there, so they cover the pods the router reads.
*/}}
{{- define "smg.discoveryNamespace" -}}
{{- if eq (.Values.router.discovery.provider | default "") "kubernetes" -}}
{{- .Values.router.discovery.kubernetes.namespace | default .Release.Namespace -}}
{{- else if .Values.router.serviceDiscovery.enabled -}}
{{- .Values.router.serviceDiscovery.namespace | default .Release.Namespace -}}
{{- else -}}
{{- .Release.Namespace -}}
{{- end -}}
{{- end }}

{{/*
Discovery file path -- the manifest the file provider reads: file.path, else
the ConfigMap's key under /etc/smg/discovery, where the router pod mounts it.
*/}}
{{- define "smg.discoveryFilePath" -}}
{{- with .Values.router.discovery.file -}}
{{- .path | default (printf "/etc/smg/discovery/%s" (.configMap.key | default "workers.json")) -}}
{{- end -}}
{{- end }}

{{/*
Discovery validation -- refuses at render time what the gateway would refuse
at start-up, or run without telling: two spellings of the worker provider,
mesh router discovery the provider cannot carry, and a provider without what
it cannot start without. Renders nothing.
*/}}
{{- define "smg.validateDiscovery" -}}
{{- $discovery := .Values.router.discovery -}}
{{- $provider := $discovery.provider | default "" -}}
{{- if and .Values.router.serviceDiscovery.enabled $provider -}}
{{- fail (printf "router.serviceDiscovery.enabled is the legacy spelling of router.discovery.provider=kubernetes and cannot be combined with router.discovery.provider=%s: set one or the other (router.discovery.kubernetes takes the serviceDiscovery settings)" $provider) -}}
{{- end -}}
{{- if include "smg.meshRouterDiscovery" . -}}
{{- if and $provider (ne $provider "kubernetes") -}}
{{- fail (printf "router.mesh without router.mesh.peerUrls finds its peers through Kubernetes discovery, which the gateway runs only under the kubernetes worker provider: with router.discovery.provider=%s, set router.mesh.peerUrls" $provider) -}}
{{- end -}}
{{- /* Router discovery watches the namespace worker discovery watches, and
     the router pods are in the release's: another namespace finds no peer,
     and every namespace finds the routers of other releases. */}}
{{- if and (eq $provider "kubernetes") (or $discovery.kubernetes.clusterWide (ne ($discovery.kubernetes.namespace | default .Release.Namespace) .Release.Namespace)) -}}
{{- fail (printf "router.mesh without router.mesh.peerUrls finds its peers in the namespace Kubernetes discovery watches, so router.discovery.kubernetes must watch the release namespace %s (no clusterWide, no other namespace): otherwise set router.mesh.peerUrls" .Release.Namespace) -}}
{{- end -}}
{{- end -}}
{{- if eq $provider "file" -}}
{{- if not (or $discovery.file.configMap.name $discovery.file.path) -}}
{{- fail "router.discovery.provider=file needs its manifest: router.discovery.file.configMap.name (an existing ConfigMap) or router.discovery.file.path" -}}
{{- end -}}
{{- else if eq $provider "slurm" -}}
{{- with $discovery.slurm -}}
{{- if not .port -}}
{{- fail "router.discovery.provider=slurm needs router.discovery.slurm.port, the inference server's port on every node" -}}
{{- end -}}
{{- if not (or .jobNames .encodeJobNames .prefillJobNames .decodeJobNames) -}}
{{- fail "router.discovery.provider=slurm needs at least one job name: router.discovery.slurm.jobNames, encodeJobNames, prefillJobNames or decodeJobNames" -}}
{{- end -}}
{{- end -}}
{{- else if eq $provider "consul" -}}
{{- with $discovery.consul -}}
{{- if not .service -}}
{{- fail "router.discovery.provider=consul needs router.discovery.consul.service, the Consul service whose instances are the workers" -}}
{{- end -}}
{{- if .tls.existingSecret -}}
{{- $keys := .tls.secretKeys | default dict -}}
{{- if ne (empty $keys.clientCert) (empty $keys.clientKey) -}}
{{- fail "router.discovery.consul.tls.secretKeys.clientCert and clientKey go together: set both or neither" -}}
{{- end -}}
{{- if not (or $keys.caCert $keys.clientCert) -}}
{{- fail "router.discovery.consul.tls.existingSecret is mounted for no file: set secretKeys.caCert, or clientCert and clientKey" -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- end }}

{{/*
Consul TLS files -- the files of router.discovery.consul.tls.existingSecret
the router pod mounts under /etc/smg/consul-tls, as a JSON list for
fromJsonArray: each with its key within the Secret (key), its file name in
the mount (path) and the Consul variable naming it (env). Empty unless the
consul provider has a TLS Secret.
*/}}
{{- define "smg.consulTlsFiles" -}}
{{- $files := list -}}
{{- $consul := .Values.router.discovery.consul -}}
{{- if and (eq (.Values.router.discovery.provider | default "") "consul") $consul.tls.existingSecret -}}
{{- $keys := $consul.tls.secretKeys | default dict -}}
{{- range $file := list (dict "key" $keys.caCert "path" "ca.crt" "env" "CONSUL_CACERT") (dict "key" $keys.clientCert "path" "client.crt" "env" "CONSUL_CLIENT_CERT") (dict "key" $keys.clientKey "path" "client.key" "env" "CONSUL_CLIENT_KEY") -}}
{{- if $file.key -}}
{{- $files = append $files $file -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- toJson $files -}}
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
{{- else if eq (.Values.router.discovery.provider | default "") "kubernetes" }}
{{- /* The Kubernetes worker provider is on (below) and router discovery runs
     inside it, in its namespace (the release's, smg.validateDiscovery), so
     only the router selector is added; --service-discovery would conflict
     with --discovery-provider. The selector names the release, so another
     release's routers in the namespace are not taken for peers. */}}
- "--router-selector"
- "app.kubernetes.io/component=router"
- "app.kubernetes.io/instance={{ .Release.Name }}"
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
{{- else if and .Values.workers (not (include "smg.discoveryProvider" .)) (not (include "smg.meshRouterDiscovery" .)) }}
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
{{- with .Values.router.discovery }}
{{- if .provider }}
- "--discovery-provider"
- {{ .provider | quote }}
{{- if eq .provider "kubernetes" }}
{{- with .kubernetes }}
{{- if .selector }}
- "--selector"
- {{ .selector | quote }}
{{- end }}
- "--service-discovery-port"
- {{ .port | quote }}
{{- if not .clusterWide }}
- "--service-discovery-namespace"
- {{ .namespace | default $.Release.Namespace | quote }}
{{- end }}
{{- if .modelIdFrom }}
- "--model-id-from"
- {{ .modelIdFrom | quote }}
{{- end }}
{{- end }}
{{- else if eq .provider "file" }}
- "--discovery-file"
- {{ include "smg.discoveryFilePath" $ | quote }}
{{- else if eq .provider "slurm" }}
{{- with .slurm }}
{{- range .jobNames }}
- "--slurm-job-name"
- {{ . | quote }}
{{- end }}
{{- range .encodeJobNames }}
- "--slurm-encode-job-name"
- {{ . | quote }}
{{- end }}
{{- range .prefillJobNames }}
- "--slurm-prefill-job-name"
- {{ . | quote }}
{{- end }}
{{- range .decodeJobNames }}
- "--slurm-decode-job-name"
- {{ . | quote }}
{{- end }}
- "--slurm-port"
- {{ .port | quote }}
{{- if .bootstrapPort }}
- "--slurm-bootstrap-port"
- {{ .bootstrapPort | quote }}
{{- end }}
{{- if .user }}
- "--slurm-user"
- {{ .user | quote }}
{{- end }}
{{- if .account }}
- "--slurm-account"
- {{ .account | quote }}
{{- end }}
{{- if .partition }}
- "--slurm-partition"
- {{ .partition | quote }}
{{- end }}
{{- end }}
{{- else if eq .provider "consul" }}
{{- /* The ACL token and the TLS files are Consul's environment variables,
     set on the router container from Secrets, never flags. */}}
{{- with .consul }}
- "--consul-service"
- {{ .service | quote }}
{{- if .address }}
- "--consul-address"
- {{ .address | quote }}
{{- end }}
{{- if .datacenter }}
- "--consul-datacenter"
- {{ .datacenter | quote }}
{{- end }}
{{- end }}
{{- end }}
{{- end }}
{{- end }}
{{- /* Under either spelling of Kubernetes discovery, as under any provider. */}}
{{- if and (include "smg.discoveryProvider" .) .Values.router.discovery.checkIntervalSecs }}
- "--discovery-check-interval-secs"
- {{ int .Values.router.discovery.checkIntervalSecs | quote }}
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
{{- if .Values.router.grpcRequestTimeoutSecs }}
- "--grpc-request-timeout-secs"
- {{ .Values.router.grpcRequestTimeoutSecs | quote }}
{{- end }}
- "--shutdown-grace-period-secs"
- {{ .Values.router.shutdownGracePeriodSeconds | quote }}
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
- "--postgres-pool-max"
- {{ .Values.history.postgres.poolMax | quote }}
{{- end }}
{{- if eq .Values.history.backend "redis" }}
- "--redis-pool-max"
- {{ .Values.history.redis.poolMax | quote }}
{{- end }}
{{- if eq .Values.history.backend "oracle" }}
- "--oracle-pool-max"
- {{ .Values.history.oracle.poolMax | quote }}
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
{{- if include "smg.authSecretName" . }}
- "--api-key-file"
- {{ printf "/etc/smg/auth/%s" (include "smg.authSecretKey" .) | quote }}
{{- end }}
{{- range .Values.router.extraArgs }}
- {{ . | quote }}
{{- end }}
{{- end }}

{{/*
Auth secret -- the Secret holding the shared API key, mounted into the router
pod as a file the gateway reads with --api-key-file; the key itself is never
in the pod's arguments. `auth.existingSecret` names a Secret the operator
manages; otherwise `auth.apiKey`, when set, is written into a chart-managed
Secret. Empty when no key is configured.
*/}}
{{- define "smg.authSecretName" -}}
{{- if .Values.auth.existingSecret -}}
{{- .Values.auth.existingSecret -}}
{{- else if .Values.auth.apiKey -}}
{{- printf "%s-auth" (include "smg.fullname" .) -}}
{{- end -}}
{{- end }}

{{/*
Auth secret key -- the key within the auth Secret that holds the API key.
*/}}
{{- define "smg.authSecretKey" -}}
{{- .Values.auth.apiKeySecretKey | default "api-key" -}}
{{- end }}

{{/*
History store -- the store the history backend uses, postgres or redis (its
values are .Values.history.<store>); empty for the other backends.
*/}}
{{- define "smg.historyStore" -}}
{{- if or (eq .Values.history.backend "postgres") (eq .Values.history.backend "redis") -}}
{{- .Values.history.backend -}}
{{- end -}}
{{- end }}

{{/*
History secret -- the Secret holding the history store's connection URL,
which carries the database password. The URL reaches the router pod as the
environment variable the gateway reads (SMG_POSTGRES_DB_URL or
SMG_REDIS_URL); it is never in the pod's arguments. The store's
`existingSecret` names a Secret the operator manages; otherwise its `url`,
when set, is written into the chart-managed Secret <fullname>-history.
Empty when no URL is configured or the backend has no store.
*/}}
{{- define "smg.historySecretName" -}}
{{- with include "smg.historyStore" . -}}
{{- $store := index $.Values.history . -}}
{{- if $store.existingSecret -}}
{{- $store.existingSecret -}}
{{- else if $store.url -}}
{{- printf "%s-history" (include "smg.fullname" $) -}}
{{- end -}}
{{- end -}}
{{- end }}

{{/*
History secret key -- the key within the history Secret that holds the URL:
the store's `secretKey`, else "<store>-url".
*/}}
{{- define "smg.historySecretKey" -}}
{{- with include "smg.historyStore" . -}}
{{- (index $.Values.history .).secretKey | default (printf "%s-url" .) -}}
{{- end -}}
{{- end }}

{{/*
History URL variable -- the environment variable the gateway reads the
history store's connection URL from.
*/}}
{{- define "smg.historyUrlEnv" -}}
{{- with include "smg.historyStore" . -}}
{{- if eq . "postgres" }}SMG_POSTGRES_DB_URL{{ else }}SMG_REDIS_URL{{ end -}}
{{- end -}}
{{- end }}

{{/*
History secret managed -- "true" when the chart writes the history Secret
itself (the store's `url` is set and no `existingSecret` replaces it). The
router pod template carries a checksum of that Secret, so a changed URL
rolls the pods.
*/}}
{{- define "smg.historySecretManaged" -}}
{{- with include "smg.historyStore" . -}}
{{- $store := index $.Values.history . -}}
{{- if and $store.url (not $store.existingSecret) -}}true{{- end -}}
{{- end -}}
{{- end }}

{{/*
Oracle secret -- the Secret holding the Oracle connect descriptor, database
user and password. Each reaches the router pod as the environment variable
the gateway reads (ATP_DSN, ATP_USER, ATP_PASSWORD); none is in the pod's
arguments. `history.oracle.existingSecret` names a Secret the operator
manages; otherwise the credentials set under history.oracle are written into
the chart-managed Secret <fullname>-secrets, the Secret earlier chart
versions wrote without reading it (same name, same keys). Empty when the
backend is not oracle or no credential is configured.
*/}}
{{- define "smg.oracleSecretName" -}}
{{- if eq .Values.history.backend "oracle" -}}
{{- with .Values.history.oracle -}}
{{- if .existingSecret -}}
{{- .existingSecret -}}
{{- else if or .dsn .user .password -}}
{{- printf "%s-secrets" (include "smg.fullname" $) -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- end }}

{{/*
Oracle secret managed -- "true" when the chart writes the Oracle Secret
itself (a credential is set under history.oracle and no `existingSecret`
replaces them). The router pod template carries a checksum of that Secret,
so a changed credential rolls the pods.
*/}}
{{- define "smg.oracleSecretManaged" -}}
{{- if eq .Values.history.backend "oracle" -}}
{{- with .Values.history.oracle -}}
{{- if and (not .existingSecret) (or .dsn .user .password) -}}true{{- end -}}
{{- end -}}
{{- end -}}
{{- end }}

{{/*
Oracle credentials -- the credentials the router pod is wired to, as a JSON
list for fromJsonArray: each with the environment variable the gateway reads
it from (env), its key within the Oracle Secret (key) and its value from the
values (value, used by the chart-managed Secret alone). With `existingSecret`
all three are wired and that Secret holds their keys; otherwise the ones set
under history.oracle. Empty when there is no Oracle Secret.
*/}}
{{- define "smg.oracleCredentials" -}}
{{- $credentials := list -}}
{{- if include "smg.oracleSecretName" . -}}
{{- $oracle := .Values.history.oracle -}}
{{- $keys := $oracle.secretKeys | default dict -}}
{{- $dsn := dict "env" "ATP_DSN" "key" ($keys.dsn | default "oracle-dsn") "value" $oracle.dsn -}}
{{- $user := dict "env" "ATP_USER" "key" ($keys.user | default "oracle-user") "value" $oracle.user -}}
{{- $password := dict "env" "ATP_PASSWORD" "key" ($keys.password | default "oracle-password") "value" $oracle.password -}}
{{- range $credential := list $dsn $user $password -}}
{{- if or $oracle.existingSecret $credential.value -}}
{{- $credentials = append $credentials $credential -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- toJson $credentials -}}
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
{{/* Engine images are published under ghcr.io/smg-project/smg (the gateway image is on docker.io) */}}
{{- $registry := $workerRegistry | default "ghcr.io" -}}
{{- $repository := $workerRepository | default "smg-project/smg" -}}
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
Worker metrics port -- the port the engine pod serves Prometheus metrics on,
or empty when it serves none. A gRPC worker with `metrics.enabled` runs the
engine behind the Rust servicer, whose own listener answers `GET /metrics`
with the `smg_servicer_*` families (requests by finish reason, refusals,
errors, in-flight requests, token totals, the first-token / inter-token /
end-to-end latency histograms, the servicer's lifecycle state); the Python
servicer has no endpoint. An HTTP worker runs the engine's own server, which
the chart leaves as it is.
Expects a dict with keys: worker
*/}}
{{- define "smg.workerMetricsPort" -}}
{{- $mode := .worker.connectionMode | default "http" -}}
{{- if and (eq $mode "grpc") (.worker.metrics).enabled -}}
{{- (.worker.metrics).port | default 29100 -}}
{{- end -}}
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
