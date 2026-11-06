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
identifiers, recorded settings, source/skill/runtime hashes, metrics, evidence
links and bounded, hash-verified candidate Python source. Prompts, native session traces, raw provider payloads and credentials are
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
distinct performance metrics and preservation of the previous snapshot.

## Public publication qualification (2026-11-06)

The [public page](https://aloschilov.github.io/pyasc-skill-stack/) was refreshed
from `adb5f9773e8bc179bd648145ad697da4ae7fcdfc` by
[run 42215507117](https://github.com/aloschilov/pyasc-skill-stack/actions/runs/42215507117).
All five retained records were available, including the complete DeepSeek/GLM
comparison. Public HTML/CSS/JavaScript/JSON bytes matched the Pages artifact.
Eighteen offline tests passed; test/build/deploy took 11/19/71 seconds. Chrome
verified all five cards, including the model denominators and four intervals.

The real model archive exposed a projection bug: request settings were validated
without the original receipt's token cap. The adapter now checks each original
receipt before deduplicating settings for display. A regression reproduces the
old rejection and checks missing, mismatched and boolean caps. No qualification
report, candidate or measured result was changed.

The [durable publication archive](https://github.com/aloschilov/pyasc-skill-stack/releases/tag/qualification-dashboard-42215507117)
contains the sources, public snapshot, log and hash receipts. Archive SHA256:
`dd913ebe83dec958d31c3d2f3c8750b751887f944dabb79340f0d19418f233a7`.
Its 23 indexed files were verified and copied to the external Mac/VM stores.
The temporary PR-branch deployment rule was removed, leaving the original
`main` rule intact. The earlier four-record publication remains in its own
[retained release](https://github.com/aloschilov/pyasc-skill-stack/releases/tag/qualification-dashboard-42205845893).

This qualifies publication of the retained scopes, not a new generation or
numerical run. Manual fixture results remain explicitly manual; their later CI
checks are not presented as the origin of those numerics. Refresh remains manual.

## Visual explorer (issue #42)

The presentation uses the compact summaries, progress indicators and matrix from
[the July 16 generator at f7bed297](https://github.com/aloschilov/pyasc-skill-stack/blob/f7bed2970d6649114b8da295cbfe2c7db7fc95c9/tests/tools/generate_dashboard.py)
as its design reference. That generator and its contemporaneous capability and
evidence files were rendered locally before choosing this layout. The reconstruction
is a visual reference, not evidence of deployment or renewed qualification of its
historical claims. Its old tiers, performance results and run history are not
imported into the current page.

The previous presentation at `ab28634` put five large evidence cards in sequence.
The explorer instead provides catalog record counts, a results matrix, task search,
origin/model/verification/outcome filters, paired interval plots and a separate
hardware metric panel. All filters select **whole records**: choosing one model or
task never silently narrows a cohort denominator. The overview always describes
the full snapshot. Observed and requested model identities are labelled separately.
Progress bars use each record's own denominator; fixture exclusions are not failures
or numerical checks. Graphs share a −100 to +100 percentage-point scale and retain
zero, negative bounds, pair counts, conditional intervals and cohort failures.
Arm-level rates are not inferred from aggregate deltas.

The matrix scrolls horizontally on narrow screens, with a visible hint and a
keyboard-focusable region. Native labelled controls, visible focus indicators and
text status labels do not depend on color. Scope/provenance opens a native modal
with Escape dismissal and focus return; its JSON is also keyboard-scrollable.
All dynamic strings use text nodes. The existing CSP and GitHub HTTPS link checks
remain in place. Missing or invalid summary data clears every result/plot and
disables filters. The archive parser, catalog, source pins, published counts and qualification
contracts are unchanged. The projection allowlist now additionally permits
explicitly selected, bounded, hash-verified candidate Python source.

No comparable history series, reference-kernel coverage or historical tiers are
available in this catalog, so those boundaries have explicit empty-state text.
Incomplete/unavailable comparisons do not show effects; unmatched NPU filters do
not inherit simulator timing. Baseline/candidate geometric mean and service score
are separate, including ratios below one.

### Review and validation

The local review bundle is outside Git at
`/Users/aloschilov/articles-workspace/dashboard-review-42/`:

- `index.html`: human review entry point, with historical, before/after desktop
  and mobile screenshots plus synthetic empty/stale/unavailable/error views.
- `historical-source/` and `historical/`: July reference inputs and reconstruction.
- `before-published/` and `after/`: old/new UI with byte-identical published JSON.
- `before/`: fresh GET-only validation of all five pinned archives; its records
  equal the published records exactly (build timestamp differs).
- `verification.json`, `SHA256SUMS` and `capture.cjs`: rendered count/provenance,
  accessibility and screenshot receipts. No run logs or archives enter Git.

Desktop (1440 px) and mobile (390 px) screenshots were visually inspected. The
published record counts remain 5/24, 11/24, 1/1, 6/1155 and 20/20 respectively;
all outcome counts and full provenance objects were compared, not just totals.
The published page is the input to this review, not a deployment of the redesign.

Run the read-only offline projection tests and synthetic browser checks:

```sh
python3 -m unittest discover -s tests/unit/python -p test_evidence_dashboard.py
npm ci --prefix tests/browser
(cd tests/browser && npx playwright install chromium && npm test)
# Use an already installed Chrome instead, if desired:
PLAYWRIGHT_CHANNEL=chrome npm test --prefix tests/browser
```

The ten browser checks cover combined filters and unchanged denominators,
keyboard interaction and focus return, explicit empty/stale/unavailable views,
malformed data/unsafe links, escaped strings, intervals/timing and responsive
layouts at 1440/768/390/320 px. axe-core checks WCAG A/AA rules including contrast
on the rendered page, modal and unavailable view. The 22 Python tests
continue to cover archive safety, public-field exclusion and evidence contracts.
The Pages test job runs both suites; it does not publish on pull requests.


### Generated kernel files

The source viewer reads only candidate paths selected by the retained lane's plan
and state. Reviewed candidates bind to the accepted cycle and verifier copy;
paired/model attempts bind to their trial plan and saved/project candidate copies;
hardware uses the original candidate from the generation record. Every displayed
file must match its recorded SHA256. Files are limited to 128 KiB of UTF-8 text,
with bounded trial/attempt counts and aggregate size. The builder never imports,
executes or broadly copies archive Python. Other files, prompts, native traces and
credentials remain excluded. Source text is rendered through `textContent`, not
HTML, and is never executed in the browser.

Source availability is independent of numerical acceptance. A missing or invalid
source projection has an explicit unavailable state; it cannot change a cohort's
qualified count or hide failed attempts. Old summaries without this extension
remain readable with an explicit source-unavailable label.

The checked public archives provide **45** files: 19 paired, 24 model-comparison,
one reviewed candidate and one original hardware candidate. Failed files retain
their failed outcome. Five paired trials exhausted their generation budgets and
have neither a candidate hash nor a retained candidate file. Their absence is
tracked in [#43](https://github.com/aloschilov/pyasc-skill-stack/issues/43); no
replacement run or backfill was performed. Upstream fixtures are explicitly
outside the generated-kernel source view.
