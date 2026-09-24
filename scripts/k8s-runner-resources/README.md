# ARC (Actions Runner Controller) Deployment Guide

This directory configures the `smg-project/smg` runner scale sets. The shared ARC
controller and listener pods run in `org-actions-runner`; runner scale-set objects,
credentials, and runner pods run in `actions-runner-system`.

The controller and runner scale-set charts are pinned to `0.14.2`. The runner
image is `fra.ocir.io/idqj093njucb/action-runner:2.337.0` for both the runner and
the `init-dind-externals` container in every active values file. Chart versions
and the runner software version are independent.

## Prerequisites

- Kubernetes cluster v1.28 or newer
- `kubectl` configured with cluster access
- Helm v3
- Organization-owner access to install and authorize the GitHub App

## 1. Create and install a GitHub App

Create a GitHub App under the `smg-project` organization with webhooks disabled.
Grant only the permissions required by ARC:

- Repository permissions:
  - **Administration**: Read and write
  - **Metadata**: Read-only
- Organization permissions:
  - **Self-hosted runners**: Read and write

Install the App on `smg-project` and grant it access to the `smg` repository. Record
the App ID and the installation ID, then generate and securely store a private key.
The installation ID is the final number in the installation settings URL:

```text
https://github.com/organizations/smg-project/settings/installations/<installation-id>
```

See GitHub's [ARC authentication documentation](https://docs.github.com/en/actions/how-tos/manage-runners/use-actions-runner-controller/authenticate-to-the-api)
for the authoritative permission list.

## 2. Create or update the Kubernetes secret

Never paste credentials into `archived/arc-secret-template.yaml` or commit a private key.
Create the secret directly from the values and PEM file instead:

```bash
kubectl create namespace actions-runner-system --dry-run=client -o yaml \
  | kubectl apply -f -

kubectl create secret generic github-arc-secret \
  --namespace actions-runner-system \
  --from-literal=github_app_id='<app-id>' \
  --from-literal=github_app_installation_id='<installation-id>' \
  --from-file=github_app_private_key='<path-to-private-key.pem>' \
  --dry-run=client -o yaml \
  | kubectl apply -f -
```

The archived `arc-secret-template.yaml` contains placeholders only and is retained
for schema/reference purposes.

## 3. Install or upgrade the shared controller

For an existing installation, review GitHub's [ARC upgrade procedure](https://docs.github.com/en/actions/how-tos/manage-runners/use-actions-runner-controller/deploy-runner-scale-sets#upgrading-arc)
before changing chart versions. [Helm does not upgrade existing CRDs](https://helm.sh/docs/chart_best_practices/custom_resource_definitions/#some-caveats-and-explanations),
so matching chart versions alone do not ensure that the installed schema is current.

```bash
helm upgrade --install arc \
  --namespace org-actions-runner \
  --create-namespace \
  --version 0.14.2 \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set-controller
```

Verify the controller:

```bash
kubectl get pods \
  --namespace org-actions-runner \
  -l app.kubernetes.io/part-of=gha-rs-controller
```

### Check the EphemeralRunnerSet schema

ARC `0.14.x` needs `EphemeralRunnerSet.status.phase` to track outdated runner sets.
An older CRD can discard this field even when the controller upgrade succeeded,
producing `unknown field "status.phase"` warnings and preventing outdated sets from
retiring normally. Check that the field type is `string`:

```bash
kubectl get crd ephemeralrunnersets.actions.github.com \
  -o jsonpath='{.spec.versions[?(@.name=="v1alpha1")].schema.openAPIV3Schema.properties.status.properties.phase.type}{"\n"}'
```

If the field is missing on an existing `0.14.1` or `0.14.2` installation, the
following additive repair restores the field from the upstream schema without
deleting runner resources. It is a repair for this specific schema drift, not a
replacement for the full CRD upgrade procedure when changing ARC versions.

```bash
kubectl get crd ephemeralrunnersets.actions.github.com -o yaml \
  > /tmp/ephemeralrunnersets-crd-before.yaml

kubectl patch crd ephemeralrunnersets.actions.github.com \
  --type=json --dry-run=server \
  --patch-file scripts/k8s-runner-resources/ephemeralrunnersets-phase.patch.json

# Apply only after the dry run succeeds, then repeat the field check above.
kubectl patch crd ephemeralrunnersets.actions.github.com \
  --type=json \
  --patch-file scripts/k8s-runner-resources/ephemeralrunnersets-phase.patch.json
```

## 4. Install or upgrade runner scale sets

Each values file explicitly references the shared controller service account in
`org-actions-runner`. Install the required scale sets into `actions-runner-system`:

```bash
helm upgrade --install k8s-runner-cpu \
  --namespace actions-runner-system \
  --create-namespace \
  --version 0.14.2 \
  -f scripts/k8s-runner-resources/runner-values-cpu.yaml \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set

helm upgrade --install 1-gpu-h100 \
  --namespace actions-runner-system \
  --create-namespace \
  --version 0.14.2 \
  -f scripts/k8s-runner-resources/runner-values-1-gpu-h100.yaml \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set

helm upgrade --install 2-gpu-h100 \
  --namespace actions-runner-system \
  --create-namespace \
  --version 0.14.2 \
  -f scripts/k8s-runner-resources/runner-values-2-gpu-h100.yaml \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set

helm upgrade --install 4-gpu-h100 \
  --namespace actions-runner-system \
  --create-namespace \
  --version 0.14.2 \
  -f scripts/k8s-runner-resources/runner-values-4-gpu-h100.yaml \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set
```

## 5. Verify

```bash
# Scale-set objects and runner pods
kubectl get autoscalingrunnersets,pods --namespace actions-runner-system

# Listener pods managed by the shared controller
kubectl get pods \
  --namespace org-actions-runner \
  -l actions.github.com/scale-set-name
```

Each scale set should have a listener pod in `Running` state. Runner pods are created
on demand when a workflow uses the corresponding `runnerScaleSetName` as its
`runs-on` label.

## Maintaining the runner image

The [Dockerfile](Dockerfile) updates the runner runtime and bundled Node runtimes
on top of the existing `v0.0.3` image, pinned by digest. It preserves the custom
startup scripts, container hooks, Docker client, and other installed tools, and
verifies the runner archive's SHA-256 before extracting it.

The published `2.337.0` image digest is
`sha256:df427c441ea192d3129d9f2e206ade6bb5e03f41b44c7d745f7a02351e5164fd`.
To build and validate an image from the repository root with OCIR access:

```bash
docker build --platform linux/amd64 \
  -f scripts/k8s-runner-resources/Dockerfile \
  -t fra.ocir.io/idqj093njucb/action-runner:2.337.0 \
  scripts/k8s-runner-resources

docker run --rm --network none \
  --entrypoint /home/runner/bin/Runner.Listener \
  fra.ocir.io/idqj093njucb/action-runner:2.337.0 --version
```

For a future runner release, update `RUNNER_VERSION` and `RUNNER_SHA256` in the
Dockerfile using the matching Linux x64 archive from the [official runner release](https://github.com/actions/runner/releases),
and use that version as the image tag. After validation, push that new tag to OCIR
and update **both** image references in all four active `runner-values-*.yaml`
files. Keep published tags immutable; use a revision suffix if rebuilding the
same runner version with different image contents.

If runners report `Runner version ... is deprecated and cannot receive messages`,
refresh the image and redeploy the scale sets. If GPU pods remain pending despite
free GPUs, also inspect Volcano PodGroups and outdated EphemeralRunners:

```bash
kubectl get podgroups.scheduling.volcano.sh,ephemeralrunners,pods \
  --namespace actions-runner-system
```

PodGroups left by outdated runners can reserve queue capacity after their pods
are gone. Repair the schema and update the image first. Any remaining cleanup
must be limited to the affected scale sets and to outdated owners with no pods;
retain reservations for active jobs and other workloads.

## Uninstalling

Remove scale sets before removing the shared controller:

```bash
helm uninstall <runner-set-name> --namespace actions-runner-system
helm uninstall arc --namespace org-actions-runner
```

Do not uninstall the shared controller until every runner scale set that uses it has
been removed.

## Archived manifests

Inactive scale-set values and deprecated `actions.summerwind.dev` resources are in
[`archived/`](archived/README.md). They are retained only for historical recovery
and must not be applied alongside the active runner scale sets.
