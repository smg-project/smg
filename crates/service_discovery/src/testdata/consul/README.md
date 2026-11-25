# Consul recordings

Real answers of the health endpoint the Consul provider queries,
`GET /v1/health/service/smg-worker?passing=true`, recorded from a dev-mode
agent (`consul agent -dev`) per release in a throwaway container: the official
`hashicorp/consul` images 2.0.4 and 1.22.7. The JSON was re-serialized
compactly; its content is as Consul answered.

The agent's own node is `agent-node` (address `127.0.0.1`); the others were
registered through the catalog API, as nodes without an agent are. Every node
is in datacenter `dc1`.

| Node | Node address | Service ID | Service address | Port | Metadata | Checks |
| --- | --- | --- | --- | --- | --- | --- |
| `agent-node` | `127.0.0.1` | `prefill-1` | `10.0.0.21` | 8000 | every `smg_` key | a TTL check, passing |
| `agent-node` | `127.0.0.1` | `decode-1` | none | 8001 | `smg_worker_type=decode` | a TTL check, passing |
| `agent-node` | `127.0.0.1` | `decode-2` | `10.0.0.23` | 8000 | `smg_worker_type=decode` | a TTL check, critical: absent |
| `agent-node` | `127.0.0.1` | `decode-3` | `10.0.0.24` | 8000 | none | a TTL check, warning: absent |
| `gpu-07` | `10.0.1.7` | `worker-a` | none | 8000 | `smg_worker_type=decode` | passing |
| `gpu-08` | `fd00::8` | `worker-a` | none | 8000 | `smg_worker_type=decode` | passing |
| `gpu-09` | `10.0.1.9` | `worker-b` | none | 8000 | none | none at all |

A service named `not-smg` on `gpu-07` is in neither recording.

What the recordings showed, and the provider relies on:

- Both releases answer in the same shape. `passing=true` keeps only instances
  whose checks all pass, warning excluded; an instance with no checks at all
  passes.
- A node registered through the catalog has an empty `Node.ID`, so the
  provider names nodes by `Node.Node`. Service IDs repeat across nodes.
- A service registered with no address has `Service.Address` empty; the
  node's address stands in. An IPv6 address comes unbracketed.
- `Service.Meta` is `null` for a service registered with no metadata.
- Every answer carries `X-Consul-Index`. A blocking query that times out
  answers in full with the index it was given; one for a service that does
  not exist answers `[]`, with an index.
- An unknown datacenter is `500 No path to datacenter`; an unknown ACL token
  is `403 ACL not found`.
- With ACLs on and `default_policy = "deny"` (`X-Consul-Default-Acl-Policy:
  deny`), a read with no token answers `200 []` and nothing more, like an
  empty service. A token with `service:read` but no `node:read` answers
  `200 []` with `X-Consul-Results-Filtered-By-Acls: true`. A token with both
  reads the instances.
