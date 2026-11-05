# Shepherd Model Gateway v{{version}}

<!-- AUTHOR: the opening paragraph: what this release makes faster, lighter or safer, with the two or three headline numbers in bold and the one new capability; name the sections a reader should go to first. Replace this comment with the paragraph. -->

This release includes {{commits}} commits from {{contributors}} contributors<!-- AUTHOR: , including N first-time contributors (GitHub's generated notes name them) -->. <!-- AUTHOR: when Upgrade notes is not empty: "Some defaults changed, and some deployments need a step before upgrading: see [Before you upgrade](#before-you-upgrade)." -->

<!-- AUTHOR: where the figures come from (this release's validation run, benchmarks of the individual changes), that they measure different workloads and are not to be combined, and what p99 means. -->

## Before you upgrade

<!-- AUTHOR: one bullet per step a deployment takes before or during the upgrade (a schema migration, a flag that keeps the old behaviour, an upgrade order between gateways and engine pods), bold subject first, each pointing at its subsection under Upgrade notes. Write them from the candidates under Upgrade notes; replace this comment. -->

## Highlights

<!-- AUTHOR: a highlight is a "### <what the user gets>" heading and a short narrative: the behaviour, and the numbers where they are known (a table for a validation run, the before -> after figures of a benchmark). The candidates below are the release-note sentences of the feature and performance pull requests, by the area of their subject's scope: merge, reword and reorder them into narratives and drop what is not a highlight. A line marked as having no release note carries the pull request's subject instead. Replace this comment and the candidates with the narratives. -->

{{highlights}}

### How this release was validated

<!-- AUTHOR: the verification suites, the failure drills, the soaks and the parity sets the release build ran through, with their numbers; what regressed against the previous candidate and what did not. -->

## Notable fixes

These bugs shipped in v{{previous}}. Problems introduced and fixed during this cycle are not listed.

<!-- AUTHOR: a fix is a bold symptom and what happens now, grouped by area ("### Security" first when there is one). The candidates below are the release-note sentences of the fix pull requests, by area: keep the ones a v{{previous}} user could have met, drop the ones that fixed this cycle's own regressions. Replace this comment and the candidates. -->

{{notable_fixes}}

## Upgrade notes

<!-- AUTHOR: the candidates below come from pull requests that declared a breaking change, a removed or renamed flag or setting, a changed default or a configuration that now refuses to start; each sits under the subsection its declaration suggests, with the declaration named. A Storage subsection opens the list only when the release carries a schema migration (the tool opens it when a pull request declares a storage change). Check every candidate against its diff, add what the pull requests did not declare, delete the subsections that stay empty, then replace this comment. -->

{{upgrade_storage}}

### Configuration that now refuses to start

{{upgrade_refuses_to_start}}

### Changed defaults

{{upgrade_defaults}}

### API behavior

{{upgrade_api}}

### Metrics

{{upgrade_metrics}}

### Engine side

{{upgrade_engine}}

### Bindings and Rust crates

{{upgrade_bindings}}

{{upgrade_unsorted}}

## Packages

{{packages}}

## Known limits

<!-- AUTHOR: what is opt-in or in beta, what a configuration cannot do yet, the parity cases that still differ from the reference, with the numbers. -->

<!-- AUTHOR: the thanks, and the first-time contributors by handle. -->

{{full_changelog}}

## What's Changed

{{whats_changed}}
