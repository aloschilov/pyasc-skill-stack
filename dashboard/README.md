# Public evidence dashboard

This read-only page projects retained qualification reports. `sources.json`
contains only reviewed release asset references, SHA256 values and CI identities;
it does not duplicate measured outcomes. Generated summaries and page snapshots
remain external artifacts. No model, compiler, simulator or NPU is invoked.

```sh
python3 scripts/evidence_dashboard.py --output /outside/new-page-snapshot
python3 -m http.server 8765 --directory /outside/new-page-snapshot
```

The builder performs unauthenticated GET requests for the pinned public assets.
It checks archive checksums, bounded member paths, relevant indexes, plan/report
identity and internally consistent counts. It reads JSON and bytes without
extracting files or importing/executing archive code. This is a projection of
the lane's retained qualification, not an independent numerical revalidation.
Detailed candidate/session validation remains in the original lane.

The current adapters support paired/model and reviewed simulator archives,
the initial upstream target census, and the generated CANNBench qualification.
Each adapter declares its accepted record schema. Unsupported, missing, malformed
or contradictory records become **unavailable**, never an inherited green result.
An unsupported new archive format needs a reviewed adapter update.

The public JSON is an allowlisted projection: tasks, counts, timestamps, model
identifiers, recorded settings, source/skill/runtime hashes, metrics and evidence
links. Prompts, native session traces, raw provider payloads and credentials are
not copied to the site. Only intentionally public archives belong in the catalog;
adding a private service response or an automatic latest-artifact search would
change this contract. Archive code is never a deployment dependency.

Generated candidates, upstream fixtures and NPU measurements have distinct
labels and denominators. Manual target execution has no CI execution identity;
it is explicitly marked manual rather than attributing its numerics to a later
PR gate. Reference kernels have no catalog entry and are not included in the
generated totals. There is no aggregate pass percentage across different lanes.
Incomplete paired experiments suppress effect estimates. Complete paired
effects retain their fixed-task uncertainty interval; they do not establish
general skill benefit. CANNBench service score and baseline/candidate time ratio
remain separate, including ratios below one.

`Evidence dashboard` runs offline tests on relevant PRs. Its manual dispatch
builds a fresh snapshot; `publish=true` deploys that artifact to GitHub Pages.
GitHub Pages must use the Actions build type, with the intended deployment branch
permitted in the `github-pages` environment. No cron, model credentials or device
access is configured. The default manual action builds without deploying.

Every build re-fetches and verifies each archive. A deleted or changed archive
therefore becomes unavailable. This is not a continuous availability monitor:
the page displays its check timestamp, marks results older than the catalog's
192-hour threshold stale, and warns after 24 hours without a page rebuild.
Browser load failure clears all cards and displays an unavailable notice.

Builds write to a new external directory and refuse overwrite. A failed build
does not replace the deployed snapshot, whose age remains visible. Rollback
means manually rebuilding a previously reviewed Git revision/catalog and
deploying that newly validated snapshot; never copy an old green summary without
rechecking its evidence. Previous Pages deployment artifacts can be retained for
diagnosis, but they are not the sole qualification archive.

Local tests cover unavailable/malformed/checksum-invalid evidence, private-field
exclusion, unsafe archives, duplicate JSON, contradictory coverage, stale results,
distinct performance metrics and preservation of the previous snapshot. A local
browser check has verified rendering and the fixture filter. Public deployment
qualification is still pending.
