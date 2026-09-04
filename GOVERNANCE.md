# SMG Project Governance

This document describes how the Shepherd Model Gateway (SMG) project is
governed. It follows the spirit of the Linux Foundation's
[Minimum Viable Governance](https://github.com/github/MVG) framework.

## Roles

**Contributors** — Anyone who files issues, participates in discussions, or
submits pull requests. No special status is required; see
[CONTRIBUTING.md](CONTRIBUTING.md) for how to get started.

**Code Owners** — Contributors with sustained, high-quality contributions to
a specific area of the codebase (e.g., a workspace crate, bindings, CI).
Code owners review and approve changes in their areas. The authoritative
list of who owns what is
[`.github/CODEOWNERS`](.github/CODEOWNERS) — this is also the right place
to look when you need to find a reviewer or contact for a given part of
the code.

**Core Maintainers** — Responsible for the overall direction, architecture,
releases, and health of the project. Core maintainers are the default
owners (`*`) in CODEOWNERS. Current core maintainers:

- Simo Lin ([@slin1237](https://github.com/slin1237))
- Chang Su ([@CatherineSue](https://github.com/CatherineSue))
- Keyang Ru ([@key4ng](https://github.com/key4ng))

## Decision Making

Day-to-day technical decisions are made through **lazy consensus** on pull
requests and issues. A pull request may merge when it has approval from
the relevant code owners (enforced via CODEOWNERS) and passing CI.

Significant changes — new architecture, breaking API changes, new workspace
crates, release planning, deprecations — are proposed as a GitHub issue or
design discussion and decided by consensus among the core maintainers.

Public contract changes additionally follow the [API Stability
Policy](governance/api-stability.md). That policy defines compatibility categories,
deprecation windows, version decisions, migration requirements, and the owner path
for intentional breaking changes.

If consensus cannot be reached in a reasonable time, the final decision
rests with the core maintainers. Disagreements are expected to be resolved
through discussion in the open, on GitHub.

## Becoming a Maintainer

The typical path is contributor → code owner → core maintainer:

1. A contributor with a track record of quality contributions and reviews
   in an area may be nominated by an existing maintainer to become a code
   owner for that area.
2. Code owners who demonstrate sustained ownership across the project may
   be nominated as core maintainers.

In both cases, additions require agreement from the core maintainers and
are made concrete via a pull request updating CODEOWNERS.

## Stepping Down and Removal

Maintainers and code owners may step down at any time by opening a pull
request removing themselves from CODEOWNERS. Maintainers who are inactive
for an extended period (roughly 6+ months), or who violate the
[Code of Conduct](CODE_OF_CONDUCT.md), may be removed by consensus of the
remaining core maintainers.

## Communication

Project communication happens in the open on GitHub — issues, pull
requests, and discussions in
[smg-project/smg](https://github.com/smg-project/smg). Security issues
should be reported privately to the core maintainers (see contact
information above).

## Changes to Governance

Changes to this document are proposed via pull request and require
approval from the core maintainers.
