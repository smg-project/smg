# smg-service-discovery

Worker discovery for SMG, in one crate.

**Discovery** finds the serving processes a gateway routes to, and keeps a
host's worker registry in step with them. A provider watches one source and
says which workers it wants. The reconciler compares that with the registry
and asks the host's job queue to register and to remove workers. The crate
names no gateway type: it asks its host through one trait.

What it holds:

- **Providers.** Kubernetes pods chosen by label selectors, a JSON manifest
  of workers, the nodes of running Slurm jobs, and the instances of a Consul
  service whose health checks pass.
- **The reconciler.** Which workers discovery owns, the diff against the
  registry, and what is asked of the host's job queue.
- **The host trait.** `DiscoveryHost` is what the providers ask of the
  process that hosts them: its registered workers, where a registration came
  from, what it stamps on a discovered worker's spec, and its job queue.
- **Settings.** The settings a host reads for each provider, the error a
  provider refuses settings with, and the one conversion from settings to
  the selected provider's runtime settings.
- **The directory watch.** `fs_watch` tells the file provider that its
  manifest changed. A host's own file readers share it.

What it does not hold: **router peer discovery**. Finding the other routers
of a mesh is a different concern, and it stays in the gateway.

## A host's first steps

```toml
[dependencies]
smg-service-discovery = { workspace = true, features = ["kubernetes"] }
```

A host names the providers it runs as features, and implements
`DiscoveryHost`. It resolves its settings with
`RuntimeDiscoveryConfig::from_config`, which checks the provider's settings
for the given worker roles before it resolves runtime inputs such as the
Consul environment. A host can call `DiscoveryConfig::validate` to check the
supplied settings on their own. It does not read the Kubernetes model id
source. The runtime conversion parses that source. It keeps `namespace`,
`label:<key>` and `annotation:<key>`, and refuses any other form as
`discovery.model_id_source`. It does not check a key's own syntax.

The host hands the runtime settings and itself to
`start_service_discovery`, which starts the selected provider's task. The
crate's top-level docs list its modules.

## Features

The default feature set is empty: a host names the providers it runs and
builds no other. Settings for a provider the build does not hold are refused,
in words that name the provider and its feature.

| Feature | What it adds |
|---|---|
| `kubernetes` | The Kubernetes provider, and the pod helpers a host's own pod readers share. |
| `file` | The file provider, and the manifest a host writes its registry out as. It names `fs-watch`. |
| `slurm` | The Slurm provider. |
| `consul` | The Consul provider. |
| `fs-watch` | The directory watch alone, for a host's own file readers. |
| `test-util` | For the tests of a crate that hosts discovery: a double of the host trait whose job queue records what it took and, under `kubernetes`, that provider started on a client the test hands it and a pod fixture. |
