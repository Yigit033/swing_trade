# Discovery Engine V2 — Phase 1

Audit / implementation: 10 October 2026. Champion code reference: `9b3bebf` plus
the pre-existing local worktree. No production module was edited in this phase.
The new package is opt-in and is not imported by the scanner or scheduler.

## Architecture and use

`python -m swing_trader.discovery` provides four explicit commands:

```powershell
python -m swing_trader.discovery snapshot
python -m swing_trader.discovery legacy --settings output/discovery_v2/legacy_settings.json --output output/discovery_v2/legacy-new.json
python -m swing_trader.discovery recorded-legacy --input output/discovery_v2/recorded_runs.json --run-id 131 --output output/discovery_v2/recorded-new.json
python -m swing_trader.discovery compare --snapshot output/discovery_v2/final_snapshots/20261010T201236851614Z --legacy output/discovery_v2/legacy_recorded_run131.json --output output/discovery_v2/comparison-new.json
```

Optional, offline cache inspection:

```powershell
python -m swing_trader.discovery compare --snapshot output/discovery_v2/final_snapshots/20261010T201236851614Z --legacy output/discovery_v2/legacy_recorded_run131.json --trusted-cache output/_live_data.pkl --as-of 2026-05-29 --output output/discovery_v2/cache-new.json
```

Output names are exclusive-create: choose a new filename for each run. Pickle is
only for explicitly trusted existing local research caches; never load a downloaded
or user-supplied untrusted pickle. Snapshot format itself is JSON + plain text.
The CLI never reads `.env`, connects to the production DB, or initializes schemas.
The audit's DB observations were obtained separately with read-only transactions.

- `master.py`: fetch two official HTTPS directory files; validate headers, rows,
  flags, duplicate symbols and footer; classify and preserve every raw row.
- `legacy.py`: execute the actual existing four Finviz queries and champion
  post-filters through an isolated subclass with explicit settings. Record query
  errors separately from empty successful results. Adapt sanitized scan exports.
- `comparison.py`: compare sets, aliases, exchanges, security types and exclusions;
  inspect existing OHLCV only after cutting all future rows at an explicit date.
- `__main__.py`: standalone commands, no production registration.
- `test_phase1.py`: deterministic, offline, synthetic fixture tests.

Snapshots contain raw files, retrieval time, source URL, HTTP Date/Last-Modified/
ETag, source footer, SHA-256, schema/classifier version, classifications and a
completion manifest written last. Incomplete snapshots and checksum failures are
rejected. Replay rebuilds normalized records from the raw source and compares them.
They are locally tamper-evident, not signed or externally immutable storage.

NASDAQ comes from nasdaqlisted. Only `Exchange=N` in otherlisted is NYSE;
NYSE American (`A`) and Arca (`P`) are retained as out-of-scope. Country USA is
not required in this new master: foreign issuers/ADRs remain explicitly tagged.
Official ETF/test flags take precedence. ETN/unit/warrant/preferred/right/debt/
fund/ADR/common classifications additionally use name heuristics with recorded
evidence, not an authoritative security master. Unknown records are retained in
REVIEW_REQUIRED. Class shares remain separate. Unique official aliases and
explicit class-share spelling can be compared; ambiguous symbols stay unresolved.
Yahoo symbol candidates are not claimed to be vendor-verified identifiers.

SPAC is SUSPECTED_NAME, never a confirmed fact from an acquisition-like name.
IPO date/status is UNKNOWN. `first_seen_at` is this snapshot's observation time,
not an IPO date or longitudinal first-listing assertion. Existing multiple
classes cannot be established from a class designation alone. Market cap, float,
fundamentals and cap band remain UNKNOWN, not synthetic zero or inferred values.
No market cap gate or 300-name cap is applied to the master.

Today's membership cannot establish prior listing membership. All snapshots set
`historical_point_in_time=false`. Historical cache checks also explicitly label
current-survivor bias, old researched data and the absence of a same-date control.

## Measured source and coverage results

Canonical snapshot: `output/discovery_v2/final_snapshots/20261010T201236851614Z`.
Downloaded at 2026-10-10 20:12:36 UTC (23:12:36 Istanbul). Both source footers:
`File Creation Time: 1009202621:31`; timezone is not asserted from the file.
HTTP Last-Modified: 2026-10-10 01:31:30 GMT. The initial/validated directories are
earlier classifier development artifacts; use the final snapshot (classifier v3).

| Population | NASDAQ | NYSE | Total |
|---|---:|---:|---:|
| All directory records in scope | 5,630 | 2,912 | 8,542 |
| Non-test records | 5,622 | 2,900 | 8,522 |
| Common/ADR candidate records, non-test | 3,382 | 1,863 | 5,245 |

Both raw source files contain 13,297 rows in total; 4,755 are other exchanges.
The scoped type counts below include test records; tests are separately excluded
from the candidate list. These are security records, not unique companies.

| Type | Count |
|---|---:|
| COMMON_STOCK | 4,939 |
| ADR | 318 |
| ETF | 1,369 |
| ETN | 3 |
| WARRANT | 421 |
| PREFERRED | 435 |
| UNIT | 354 |
| RIGHT | 125 |
| DEBT | 173 |
| FUND | 253 |
| UNKNOWN | 152 |

Non-test unknowns requiring review: 144. Scoped records with a class designation:
1,177. SPAC-like name flags: 749 **securities**, including units/warrants/rights;
not 749 confirmed SPAC companies. All 8,542 scoped rows have UNKNOWN market cap
and float. Thus 5,245 is a candidate superset, NOT 5,245 confirmed small/mid caps.

Live Finviz attempt: four HTTP failures; independent public endpoint check returned
403. No bypass/proxy/account rotation was attempted. Champion fallback selected
234 static names. `legacy_finviz_count=null` means unavailable, not zero discovery.
Artifacts: `legacy_20261010.json`, `comparison_live_attempt.json`.

Last recorded production scan: run 131, created 2026-10-09 02:29 UTC, bar session
2026-10-08. Rank telemetry indicates a 57-member Finviz-derived list, all 57 fetched.
Raw per-query memberships were not persisted; independent query success cannot be
recovered. All 57 are present among today's master candidates.

Descriptive difference versus this older 57-name scan:

- All non-test master securities absent: 8,465 (NASDAQ 5,598; NYSE 2,867).
- Common/ADR candidates absent: 5,188 (NASDAQ 3,358; NYSE 1,830).
- This compares different observation dates. It is not same-day recall, a count
  of missed winners, or evidence that 5,188 names satisfy cap/liquidity gates.

Seven currently listed common stocks occur in the 33-name hardcoded exclusion
set: CLOV, HOOD, LCID, OPAD, RIVN, WKHS (NASDAQ), SPIR (NYSE). Code gives a generic
"delisted/problematic" comment, no per-name rationale, expiry or evidence.
This proves potential structural exclusion, not that all seven passed current
Finviz predicates or lost profitable trades. Actual pre-exclusion counts are unknown.

Historical OHLCV cache: `output/_live_data.pkl`, 1,295 ticker keys; underlying
period 2024-06-03 through 2026-05-29. 1,190 of the 5,245 candidates have usable
55-bar windows ending May 29: **22.69%** coverage; 4,055 have no cached key.
Of the 5,188 names absent from the October 8 list, 1,140 have usable May data:
**21.97%**. This cache was originally Finviz-selected and is not representative
of the newly uncovered universe. Raw cached price/volume are revised adjusted data.

At the single historical date May 29, current trigger functions find 14 distinct
mechanical matches: VCE 4, RVOL 11, overlap 1. Thirteen pass $7–200 and $5M mean
dollar-volume gates; ACVA fails the price gate. All 14 are absent from the October
8 list, which has no meaning for their May membership. Historical market cap,
earnings and legacy membership are UNKNOWN. No signal quality/risk/exit engine
was invoked. No future returns are available after this cache endpoint; no profit
or incremental historical opportunity claim is made. At October 9, all 1,190
matched cache series are stale and usable current-bar coverage is **0%**.
Artifacts: `comparison_final.json`, `comparison_current_cache.json`.

## Verified champion chain and attrition

References: `small_cap/universe.py:get_finviz_universe/get_universe`,
`small_cap/engine.py:scan_stock/get_small_cap_universe`,
`data/fetcher.py:fetch_multiple_stocks_batch`,
`api/routers/scanner.py:_execute_smallcap_scan`.

Runtime universe settings were re-read with a read-only DB transaction: cap 300,
post-price $6–250, cache 60 minutes, static merge below 30, Finviz enabled.
Stored settings updated 2026-09-09. Code defaults are cap 260 and $7–200.
Explicit sanitized comparison settings are in `output/discovery_v2/legacy_settings.json`.

| Stage | Purpose / classification | Measured loss and uncertainty | Broad-universe treatment |
|---|---|---|---|
| Country USA | Discovery geography restriction, not exchange identity or risk | Per-predicate count unavailable | Explicit NASDAQ/NYSE; tag domicile/ADR independently |
| Cap $300M–2B / $2B–10B | Strategic size mandate | PIT/unknown fundamentals unavailable | Cap bands and UNKNOWN lane; no fabricated past cap |
| Price >$7 | Price/liquidity proxy restricting discovery | Per-predicate count unavailable | Tag price band; keep broad master |
| Avg volume >300K small / >500K mid | Coarse liquidity proxy | Per-predicate count unavailable | Measure dollar liquidity and bar availability independently |
| Q6/Q6b MA50 + 20d new high | Preselect VCE-like momentum | Conjunction, no independent denominator | Apply triggers locally after cache coverage established |
| Q7/Q7b RVOL>2 + green + MA20 | Preselect RVOL-like momentum | Same limitation | Preserve trigger, measure all eligible windows |
| Dedup | Avoid duplicate query members | Not persisted historically | Explicit sequential count in new observer |
| 33 exclusions | Generic delisted/problematic list | Seven currently listed; actual daily rejection unknown | Annotate official status; research review, no silent loss |
| Post-price $6–250 | Discovery clamp | Historical count unknown; engine still $7–200 | Report both thresholds distinctly |
| Current Volume × Price ordering | Liquidity ranking before cap | Not 20d dollar volume | Work-queue priority, never claim unprocessed names scanned |
| Cap 300 | Compute budget, not trading risk | Zero cap losses in 59 recorded runs with telemetry | No master cap; explicit processing coverage |
| Finviz <30 → static merge | Availability fallback | Current failed fetch gives 234 static names | Source provenance and stale/unresolved symbols explicit |
| Batch missing / short data | Technical availability | 68 missing observations across 60 runs | Per-symbol retry/status; not strategy rejection |
| Fetch <50% / stale majority | Aggregate data safety | Current run 57/57; per-symbol stale count not recoverable historically | Individual data eligibility, completed-session watermark |
| Engine price/cap/dollar-volume/earnings | Risk/tradability gates BEFORE triggers | `filter_failed` is pooled; separate rule attribution unavailable | Keep champion unchanged; separate shadow tags/unknowns |

Correction to the earlier audit: engine `scan_stock` applies `apply_all_filters`
BEFORE `check_all_triggers`, not after. Float and ATR are advisory at this filter
stage. Engine price is $7–200, cap $250M–10B, mean 20d dollar volume >=$5M,
earnings window 0..3 days; missing cap and earnings can pass. VCE/RVOL have their
own 55/52-bar requirements. Later swing/RSI/stage/quality rules are outside this
Phase 1 implementation and remain unchanged.

Read-only sample: last 60 stored scans, Aug 3–Oct 9, 2026 (23 distinct bar-session
values; repeated scans are NOT independent trades). 6,662 selected observations,
6,594 with data; 68 missing. 59 scans retain cap telemetry: zero cap-hit runs,
zero cut tickers, maximum post-filter universe 245. Therefore the cap is not the
observed bottleneck in this sample; earlier claims blaming 300 were unsupported.
Pooled reject observations: no_trigger 5,142; filter_failed 867; RSI 126; stage 76;
swing 32; type-quality A/C/B 64/48/14; no_data 1; scan_error 207. These span code
versions and repeated days. Aggregate fetch omissions and engine no_data counts
are different stages; do not add them as independent stocks. Error details are
not persisted here, so causes of the 207 scan errors are not established.

Latest Oct 8 session: 57 selected/fetched → 2 filter_failed → 50 no_trigger →
2 RSI rejects + 2 stage rejects + 1 swing reject → zero final signals.
This is the strongest same-run evidence; separate upstream Finviz predicate
losses cannot be reconstructed from the stored history.

## Free-provider scalability

An entire six-month redownload on every scan is unnecessary. Phase 2 should use
an independent persistent research bar store, initially populated in resumable
bounded batches. Thereafter request only missing completed sessions plus a small
overlap for revisions. Query planning must use last-valid-session per symbol,
not the last attempted date. Unknown cap/name classification must be separate
from quote-fetch status. Failures never mean delisted or no-signal.

Current fetcher loops 25-name chunks. The comment "one HTTP request per chunk"
is false: yfinance internally loops tickers and invokes per-ticker downloads;
chunking is orchestration, not a licensed bulk daily feed. 5,245 symbols imply
roughly 5,245 symbol histories, not 210 network requests. Retry only failed or
partial symbols, bounded backoff, honor Retry-After, stop on persistent 403/429;
no proxy rotation or account multiplication. A full initial universe may require
several resumable sessions; no guaranteed free full-universe SLA is established.

Current batch fallback runs only if the ENTIRE results dictionary is empty, so
partial ticker failures are not repaired. Standalone pickle research caches exist,
but production fetch_multiple_stocks_batch has no durable incremental OHLCV store.
A Yahoo timezone/HTTP cache is not a durable complete bar store.

For research persistence, store raw OHLCV, corporate-action fields, adjustment
mode, provider, retrieval time and revision checksum separately. A split/dividend
can revise the entire adjusted history: never append adjusted bars across different
adjustment bases without overlap validation. On revision, rebase/refetch only the
affected symbols, while preserving the original as-observed signal inputs. Use an
exchange session calendar including holidays/early closes. Daily bar availability
after close must be confirmed, not inferred from wall-clock alone. Do not fabricate
missing bars. Symbol renames and class aliases require dated evidence.

Provider terms were checked on October 10:

- Nasdaq Trader publishes current directory fields and source timestamps; these
  files do not provide OHLCV or a historical constituent archive.
- yfinance describes itself as a research/education tool and Yahoo data as personal
  use. Public availability is not evidence of a redistribution/commercial license.
- Tiingo's current Terms section 1.6(a) expressly disallows durable data AND derived
  product retention on Starter/Trial plans. Therefore it is not a free persistent
  cache fallback for this design. Old code comments about quotas are not proof of
  current access rights.
- Finnhub stock-candle entitlement and current quotas were not verified from the
  dynamically rendered official pricing/docs pages. Do not assume a free API key
  includes historical OHLCV. No authenticated provider requests were made.

The recommended zero-budget architecture is the official directory + a local
personal-research cache of permitted data + explicit incomplete-coverage reporting.
For a shared/commercial product, a fully free legally reusable OHLCV source is NOT
established in Phase 1. No licensing claim or rate-limit bypass is part of the code.

Sources:

- https://www.nasdaqtrader.com/trader.aspx?id=symboldirdefs
- https://www.nasdaqtrader.com/trader.aspx?id=symbollookup
- https://ranaroussi.github.io/yfinance/
- https://github.com/ranaroussi/yfinance/blob/main/yfinance/multi.py
- https://app.tiingo.com/tos/ (section 1.6)
- https://finnhub.io/docs/api/stock-candles
- https://finnhub.io/pricing

## Benchmark contract (design only; no exit-engine changes)

### A. Coverage

Freeze the universe observation available BEFORE each decision timestamp T. Keep
separate denominators: listed scope L(T), security-eligible E(T), data-valid V(T)
and actually evaluated S(T). Report data coverage |E∩V|/|E| AND evaluation coverage
|S∩E∩V|/|E∩V|. Cap eligibility must be known at T; UNKNOWN cap/type are separate
lanes with counts, not treated as ineligible or silently removed. Report cap bands
below $250M, $250–300M, $300M–2B, $2–10B, above $10B and UNKNOWN. This exposes the
existing engine/discovery boundary difference without choosing a new production
threshold. If fundamentals are unavailable, confirmed small/mid-cap coverage is
not computable; publish security-only coverage plus unknown counts.

Failures: mapping, absent history, insufficient warmup, invalid OHLCV, stale bar,
partial session, provider failure, unknown fundamentals, never scheduled. Count
each ticker once per session at each funnel stage. Show exchange/type/cap/liquidity
strata. A current directory must never substitute for an older historical snapshot.

### B. Opportunity

Before seeing outcomes, freeze same-session champion list C(T) and new candidate
list N(T), full feature/input hashes and code/config. Incremental cohort is N(T)−C(T).
If champion unavailable, label comparator unavailable; do not call all new names
incremental. Positive AND negative outcomes stay in the cohort. Research cache
results and prospective shadow results have different cohort IDs and are never pooled.

Reference entry E is the next scheduled trading session's Open. Missing/halted
entry remains NO_FILL/UNKNOWN, not a later convenient Open. At h=5/10/20/30 trading
sessions including entry day, use close of session h for R_h; maximum High over
those h sessions for MFE_h; minimum Low for MAE_h. All divide by the same E.
Report fractions/prices and percent units explicitly. MFE is an excursion, not an
achievable fill or a realized profit. No T1/T2/stop/trailing assumptions enter these
fixed-horizon diagnostics. Missing future data/delistings remain unresolved and
their rate is reported; never silently discard them from loss accounting.

R_h = 100 × (Close_h/E − 1)
MFE_h = 100 × (max(High_1..h)/E − 1)
MAE_h = 100 × (min(Low_1..h)/E − 1)
Net reference R_h = 100 × [Close_h×(1−s_sell)/(E×(1+s_buy)) − 1]

Predeclare symmetric scenario costs 5/25/50 bps per side, zero commission only
where the account actually has it; these are stress assumptions, not measured
spread/impact. Stratify on signal-time 20d dollar volume (<$5M, $5–20M, >$20M,
UNKNOWN). No claimed executable return without real manual fill/delay evidence.

Primary sampling rule: first qualifying event per ticker, suppress another event
until the full 30-session observation window ends, without looking at its outcome.
Keep all suppressed events in an audit file. Per-ticker windows are non-overlapping;
cross-ticker/calendar dependence still requires date/ticker-clustered uncertainty
and multiple-class/issuer dependence where an issuer identifier is available.
Keep cohort selection constant across horizons; show matured/missing denominators
per horizon. Publish count, mean, median, win/loss distribution, tail loss, bootstrap
intervals and best-1/3/5-excluded sensitivity. No optimization on this old research
cache; prospective shadow data remains untouched holdout.

## Validation and smallest Phase 2

Executed: 33 new deterministic tests passed; 33 existing Finviz/universe/price-band
regression tests passed (66 total). `compileall` passed. Pytest emits an existing
pytest-asyncio fixture-loop deprecation warning. Initial temp-directory setup
failed due to sandbox permissions; rerunning with an isolated project temp path
passed. No broad production suite or profitability test is claimed.

Deterministic tests cover type precedence (including unit descriptions mentioning
warrants/common shares and ADR prose mentioning a right), scope, test flags,
UNKNOWN retention, class aliases, malformed/partial source rejection, no arbitrary
cap, immutable write/replay/checksum validation, actual champion-query execution,
sequential attrition, failed-source semantics and slicing away future cache bars.

The smallest next step is an opt-in CLI research job that incrementally fills
completed daily bars for the classified candidate universe with explicit coverage
and failure states, then calls the unchanged VCE/RVOL trigger functions. Store the
same-session champion membership and all shadow candidates before outcomes mature.
No scheduler, database migration, production scanner or trading/risk/exit changes
are needed. Warmup/data validity is measured per symbol; queue limits affect work
done in a run, not master membership. Unknown security types stay in the review lane.

Phase 1 demonstrates broader auditable membership and measurable data gaps. It
does not demonstrate earlier entries, more profitable signals or improved recall.
Phase 2 has not been implemented or started.
