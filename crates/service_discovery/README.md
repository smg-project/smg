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
smg-service-discovery.workspace = true
```

A host implements `DiscoveryHost`. It resolves its settings with
`RuntimeDiscoveryConfig::from_config`, and hands the result and itself to
`start_service_discovery`, which starts the selected provider's task. The
crate's top-level docs list its modules.

## Features

The default feature set is empty.

| Feature | What it adds |
|---|---|
| `test-util` | For the tests of a crate that hosts discovery: the Kubernetes provider started on a client the test hands it, a pod fixture, and a double of the host trait whose job queue records what it took. |
