# tehillim-benchmarks

Evaluates embedding models against scholarly annotations of Psalms parallelism and genre, scoring vectors from
[tehillim-embeddings](https://github.com/rdtaylorjr/tehillim-embeddings) on a retrieval benchmark
built from aligned `parallel_*` Text-Fabric features. Consumes both as pure data dependencies,
never as code dependencies.

## Data sources

* **Parallelism structure**: `parallel_*` node features on BHSA `half_verse` nodes, loaded via
  Text-Fabric's `use("etcbc/bhsa", mod="rdtaylorjr/tehillim-logos/tf")`. No local checkout
  needed, Text-Fabric fetches and caches the module itself. The underlying annotations are
  derived from the Logos Psalms Explorer Dataset, used with permission — see Citations below.
* **Genre classification**: a genre-classification CSV (not included in this repo, supplied at
  runtime) is likewise derived from the Logos Psalms Explorer Dataset — see Citations below.
* **Embedding vectors**: Parquet files from a local `tehillim-embeddings` checkout, Hive-partitioned
  at `data/domain=semantic/model=<slug>/text=<variant>/part-0.parquet`. `--checkout` on Text-Fabric
  loaders and `embeddings_dir` on the scripts below are independent inputs.

Every benchmark script's `--output`/`--output-dir` writes into a local
[tehillim-data](https://github.com/rdtaylorjr/tehillim-data) checkout, kept as a separate repo
since result Parquet files run tens of megabytes each and bloat a code repo's clone size and
history. That checkout is Hive-partitioned the same way as `tehillim-embeddings`:
`benchmark={parallelism,genre,trajectory}/domain={lexical,semantic}/stage={raw,detail,master,shuffle_control}/...`.
`stage=raw` holds each comparison script's own CSV output, `stage=detail` the per-observation
Parquet export, `stage=master` the joined final report, `stage=shuffle_control` the order-shuffle
null control. Trajectory has no `detail`/`master`/`shuffle_control` stage: `compute_profiles.py`
writes both the per-model profile shards and the derived `trajectory_distances.parquet` together
under `stage=profiles`, `validate_against_genre.py` writes its permutation-test CSVs under
`stage=raw`, and `export_ui_rows.py` writes its JSON payloads under `stage=ui`. Point a script's
`--output`/`--output-dir` at the matching `stage=` directory. `results.csv` in the examples below
is illustrative, point it wherever you check that out.

## Methodology

`parallelism.pairs.build_retrieval_pairs` reconstructs groups from the position-aligned
`parallel_*` features, then decomposes each group's `parallel_signature` (e.g. `AB-AB-AB`) into
retrieval pairs: adjacent member-slots within one strophe segment form a pair, a dashless tricolon
like `ABC` gives overlapping pairs A-B and B-C. A pair is dropped if either slot is unresolved,
flagged ambiguous (`parallel_ambiguous`), or both slots share one node.

The primary metric is Average Precision: rank every true pair and every baseline pair by cosine
similarity and compute the area under the precision-recall curve, via
`sklearn.metrics.average_precision_score`. This is the MTEB Pair Classification protocol
(Muennighoff et al. 2023, EACL, "MTEB: Massive Text Embedding Benchmark"), chosen over AUC-ROC
because AP is robust to class imbalance while AUC-ROC reads optimistically under it. AP's chance
level is the positive-class prevalence, not 0.5, so raw AP is comparable across models within one
scope but never across scopes with different true:baseline ratios.

A separate negative control (`parallelism.baseline`) compares true-pair similarity against adjacent
colon bicola never annotated as parallel, to check whether elevated similarity is specific to
genuine parallelism or reflects generic topical continuity between neighboring lines. Results are
reported per Lowth's original typology (Lowth 1778: synonymous, antithetic, synthetic) as extended
by later scholarship (staircase/climactic, emblematic), each type reported separately.

### Statistical methodology

* **Effect size vs. z-score**: `parallelism.calibration` exposes `calibrated_z_score` (a true
  z-score, for a single observation against the background distribution) and
  `calibrated_effect_size` (a Cohen's-d-style standardized effect size, for a group mean, since
  dividing a mean by the background's population standard deviation is not a sampling-distribution
  statistic).
* **Background exclusion**: the background similarity distribution used to calibrate every effect
  size excludes nodes that participate in a true parallel pair, so the null is never partly built
  from the signal it's calibrating.
* **Multiple-comparison correction**: `parallelism.multiple_comparisons.add_fdr_q_values` adds a
  Benjamini-Hochberg `q_value` (Benjamini & Hochberg 1995) and a Benjamini-Yekutieli `q_value_by`
  (Benjamini & Yekutieli 2001, valid under arbitrary dependence, since a base model's text variants
  are correlated) next to every p-value. Correction families are keyed on
  `(source, metric, scope_kind)`, an "overall" p-value is never pooled with the per-type p-values
  that constitute it.
* **Cluster-robust permutation test**: `parallelism.retrieval_metrics.stratified_mean_gap_test`'s
  per-anchor null average excludes both the anchor's diagonal similarity and the permutation
  draw's labeled-true column, so the true signal cannot leak into the null it's compared against.
* **Cluster bootstrap with a BCa interval**: pairwise similarities sharing a source or target vector
  are correlated, so `parallelism.bootstrap.block_bootstrap_ap_gap_and_auc` resamples whole psalms,
  the natural cluster unit. AP and AUC are bounded and can be skewed near [0, 1], where a plain
  percentile interval undercovers, so the reported interval is BCa (Efron 1987) with a
  leave-one-psalm-out jackknife acceleration term, plain percentile bounds kept alongside for
  comparison.

## Genre benchmark

`src/genre` scores the same embedding models against a second, independent benchmark: does an
embedding put same-genre psalms closer together than different-genre psalms? The labels are a
third-party psalm genre classification (not included in this repo, supplied at runtime), which
labels each of the 150 psalms with exactly one of seven genres (Lament, Praise, Hymn, Royal,
Wisdom, Thanksgiving, Trust); it is a distinct data source from the `parallel_*` structure above,
though both ultimately derive from the same dataset, used with permission (see Citations below).
The two benchmarks share their embedding inputs and their statistical machinery (`src/library`).

Every one of the C(150, 2) = 11,175 psalm pairs is scored: `genre.pairs.build_genre_pairs` labels a
pair `same_genre` when both psalms carry the same genre. Each psalm's vector is the mean of
its colon embeddings (`genre.centroid.psalm_centroids`). Average Precision and AUC are
computed exactly as in the parallelism benchmark (`genre.evaluate.evaluate_genre_discrimination`),
ranking same-genre pairs against different-genre pairs by cosine similarity.

Because genre pairs are exhaustive over the full psalm population rather than a sparse annotated
subset, there is no separate baseline population to calibrate against, so the background for
effect sizes is the full 150-psalm-centroid population itself (`genre.scripts.compare_calibrated`).
The bootstrap CI (`genre.bootstrap.block_bootstrap_genre_ap_gap_and_auc`) generalizes the same
psalm-clustered BCa principle to genre's fully symmetric structure via a vertex bootstrap: resample
the 150 psalms with replacement and reconstruct the pairwise similarity/genre-match matrices from
the resampled psalms, rather than resampling the derived pairs directly.

| script | computes |
|---|---|
| `compare_models.py` | raw Average Precision (primary) and AUC, every model |
| `compare_calibrated.py` | same/different-genre calibrated effect size on top of AP/AUC |
| `compare_by_genre.py` | one-vs-rest AP/AUC per individual genre, with psalm-label permutation inference |
| `export_detail.py` | row-per-pair raw similarity and calibrated z, plus a per-model summary |
| `compute_bootstrap_cis.py` | psalm vertex-resampling BCa 95% CI on AP, gap, and AUC |
| `build_master_report.py` | joins the CSVs above into one master Parquet set with BY-FDR q-values |

```bash
.venv/bin/python -m genre.scripts.compare_calibrated \
  /path/to/genre-labels.csv /path/to/tehillim-embeddings/data/domain=semantic --output results.csv
```

### Per-genre one-vs-rest breakdown and its inference layer

The whole-population comparison above answers "genre in general vs. not," but genre labels
are heavily imbalanced (Lament 59 psalms, Praise 41, vs. Trust 6, Wisdom 9, Thanksgiving 8), so a
"genre signal detected" finding there is closer to "Lament/Praise is separable" than "genre in
general is separable." `genre.scripts.compare_by_genre` restricts to one genre at a time
(`genre.pairs.filter_pairs_by_genre`, pairs touching that genre on either side, the standard
one-vs-rest extension of a binary discrimination task) and reports AP/AUC per (model, genre).

Its first version computed significance the same way `evaluate_genre_discrimination` does: a flat
`scipy.stats.mannwhitneyu` over every pairwise psalm comparison. For a genre with few psalms and
many pairs, that overstates the effective sample size (Lament's 1,711 same-genre pairs come from
only 59 psalms, `C(59,2)=1711`), which is anti-conservative. The corrected version
(`genre.permutation.joint_psalm_label_permutation_test`) instead permutes psalm-level genre labels
(not pair rows), recomputing every genre's one-vs-rest AUC from the same permuted draw so a
Westfall-Young (1993) maxT correction is valid across the 7 genres jointly, exactly the same
psalm-level permutation principle already verified in `trajectory.scripts.validate_against_genre`.
The permuted statistic is signed (matching `evaluate_genre_discrimination`'s one-sided
`alternative="greater"` test: same-genre more similar than different-genre, not merely different in
either direction), since an unsigned `|AUC-0.5|` statistic would also credit a genre separated in
the *opposite* direction as "significant," which one real run caught directly: Hymn's naive test
and corrected permutation test disagreed sharply until the sign was fixed, because most models'
Hymn AUC sits below 0.5 (same-genre psalms less similar to each other than to other genres), a
real effect the one-sided test is designed to ignore, not detect.

Every model/genre row reports three p-values, `separation_p_naive` (the legacy flat Mann-Whitney
p, kept for reproducibility), `separation_p_perm` (the psalm-label permutation p), and
`separation_p_maxT` (the joint family-wise-corrected p), plus a jackknife BCa 95% CI on AP and AUC
(`genre.bootstrap.block_bootstrap_genre_ap_gap_and_auc`, generalized with a `population_mask`
parameter to restrict its existing psalm-vertex resampling to a one-vs-rest population). BH/BY-FDR
applies to each p-value source independently, scoped per genre. Ranking (which representation has
the highest AP for a genre) and significance (whether that separation exceeds a psalm-level
permutation null) are reported as separate questions, since a representation can rank first for a
genre descriptively while still failing to clear the permutation bar, or vice versa.

```bash
.venv/bin/python -m genre.scripts.compare_by_genre \
  /path/to/genre-labels.csv /path/to/tehillim-embeddings/data --output results.csv
```

## Lexical benchmark

`tehillim-embeddings` also ships lexical representations (`data/domain=lexical/`), built from
BHSA's lexical and surface-form features rather than a learned embedding model, across three
units: `homograph` (bare consonantal spelling, BHSA's `lex0`), `lexeme` (disambiguated
dictionary entry, BHSA's `lex`), and `word` (the inflected surface form, in `consonantal`,
`vocalized`, and `cantillation` text tiers). They score against the same parallelism and genre
benchmarks above, through the same evaluation code, no separate pipeline. Two architectural
variants exist for the positional/recurrence weightings: colon-level (each colon's vector
distinct, correct for parallelism's pairwise colon comparison) and psalm-broadcast (one
whole-psalm vector repeated across its colons, correct for genre's mean-pooled psalm centroid),
documented in `tehillim-embeddings`'s README. Parallelism-scoped UI tables exclude
`_psalm`-suffixed models (`ui_export.export._drop_psalm_level_models`), since a broadcast vector is
architecturally degenerate for a colon-pairwise task. Genre tables keep them.

### Order-shuffle-null control

Some lexical representations encode colon order directly (position-binned pyramids, lag-binned
recurrence). `library.order_shuffle.order_shuffle_result` tests whether a representation's
benchmark score reflects genuine order signal or a mechanical artifact of the binning: score the
real embeddings, score N within-psalm-order-shuffled embeddings
(`lexical.scripts.generate_shuffle_control[_colon]` in `tehillim-embeddings`), and report
`delta_order` (real score minus mean shuffled score) with a rank-based permutation p-value,
`(count(shuffled >= real) + 1) / (n + 1)`, the same convention used everywhere else in this project.
This replaced an earlier ad hoc z-score computed from a 30-draw empirical mean/std, which implied a
Gaussian-tail interpretation the sample size cannot support.

```bash
.venv/bin/python -m genre.scripts.shuffle_order_control \
  /path/to/genre-labels.csv /path/to/real_embeddings.parquet /path/to/shuffled_dir --output results.csv
.venv/bin/python -m parallelism.scripts.shuffle_order_control \
  /path/to/real_embeddings.parquet /path/to/shuffled_dir --output results.csv
```

Run against `icf_posmean_psalm` (genre) and `icf_pos4` (parallelism): genre's Hymn and Lament
deltas are individually significant (p=0.0323 each, the resolution ceiling at 30 shuffles) but do
not survive BH/BY-FDR correction across the 7 genres (q=0.1129 BH, 0.2927 BY). Parallelism's
`icf_pos4` shows a significant order effect (`delta_order=+0.1768, p=0.0323`), though the shuffle
design alone cannot distinguish a genuine colon-order signal from a bin-adjacency artifact of
the positional binning itself, an open question left unresolved by this control.

### BHSA checkout pin

`library.bhsa.DEFAULT_CHECKOUT` is pinned to `v1.8.1`, matching `tehillim-embeddings`'s local
BHSA clone, rather than floating on `"latest"`. `parallelism.tf_features.load_api` uses a separate
`_TEHILLIM_LOGOS_CHECKOUT = "v1.0"` for the `rdtaylorjr/tehillim-logos`
module, since that module has an independent release history and cannot share BHSA's pin.

`library.bhsa.load_bhsa_api` tries the local BHSA clone at `~/Developer/hebrew/bhsa/tf/2021` first,
the same clone `tehillim-embeddings` reads from. Only if that fails does it fall back to
Text-Fabric's `use()`, with a 30-second timeout (`DEFAULT_USE_TIMEOUT_SECONDS`), since `use()`
re-verifies its release against GitHub's API even when the data is fully cached locally, and can
stall or back off for minutes under a GitHub rate limit.

## Morphology benchmark

`tehillim-embeddings` also ships morphology representations (`data/domain=morphology/`),
built from BHSA's word-level grammatical features (part of speech, agreement, verbal stem/tense,
pronominal-suffix morphology) rather than lexical identity or a learned embedding model. They
score against the same parallelism and genre benchmarks above, through the same evaluation code, no
separate pipeline, with the same colon-level/psalm-broadcast split as the lexical domain
(`_psalm`-suffixed models excluded from parallelism-scoped UI tables, kept for genre). One
representation, `morph_suffix_posmean` (psalm-scale deployment), has no colon-level form at all and
isn't marked by the `_psalm` naming convention, so it's excluded from parallelism scoring entirely
rather than relying on `_drop_psalm_level_models` to catch it.

### Sparse embedding scoring

One morphology representation, `morph_signature`'s trigram construction, has 75,894 dimensions
with at most a few dozen nonzero entries per colon, and is stored sparse
(`node_id`/`indices`/`values` Parquet schema, `sparse=true` in the file's schema metadata) rather
than as a dense `vector` column, to avoid materializing a mostly-zero array per colon.
`library.embeddings.load_sparse_embeddings` reads it into a `scipy.sparse.csr_matrix`, and
`library.retrieval_metrics.sparse_cosine_similarity_matrix`,
`parallelism.evaluate.build_side_vectors_sparse`/`run_evaluation_sparse`, and
`library.centroid.sparse_psalm_centroids` with
`genre.evaluate.evaluate_genre_discrimination_sparse`/`genre.scripts.compare_by_genre.compare_model_across_genres_sparse`
score it without ever densifying the vectors, only the small model-sized similarity matrix each
produces. The standard comparison scripts (`compare_models.py`, `compare_calibrated.py`,
`compute_bootstrap_cis.py`, `compare_true_similarity.py`, both `export_detail.py` scripts) read
only the dense schema and do not dispatch to this path; scoring the sparse trigram family currently
requires calling the sparse functions directly.

## Structural trajectory analysis

`src/trajectory` asks a different kind of question than the two benchmarks above: how a psalm's
meaning moves through the poem, independent of any benchmark ranking, rather than whether an
embedding model discriminates a labeled phenomenon. It is computed for every model in a
`tehillim-embeddings` checkout, never filtered to a top-k subset, so this analysis never depends on
how a model scored elsewhere.

Every psalm gets two independent representations. The **content centroid**
(`library.centroid.psalm_centroids`, shared with the genre benchmark) is the mean of its cola
embeddings, capturing what the psalm is about. The **structural profile**
(`trajectory.self_similarity`) is its self-similarity matrix, `S[i, j] = cos(cola_i, cola_j)` over
the psalm's ordered cola, capturing how it moves through semantic space. Psalms differ in cola
count, so `resample_to_grid` maps each psalm's matrix onto a fixed `grid_size x grid_size` grid over
relative position `t = i / (n-1)`, making psalms of any length directly comparable.
`trajectory.geometry` derives three further position-indexed curves from the same ordered,
L2-normalized sequence (adjacent-cola similarity, step magnitude, turning angle between
consecutive displacements), each resampled onto the same grid.

`trajectory.distance.content_distance` (1 minus centroid cosine similarity) and
`structural_distance` (RMS difference between two resampled structural profiles) let two psalms be
compared on topic and on architecture separately.

`trajectory.scripts.validate_against_genre` is a second, independent validation of the genre
signal, deliberately apart from the AP/AUC machinery above: a permutation test of whether same-genre
psalm pairs sit closer together than different-genre pairs, on all five distance metrics. Because
genre labels correlate with psalm length in this corpus (Hymns are short, Wisdom psalms are
long and highly variable), raw distance comparisons are confounded with length. `residualize_by_length`
removes that confound via a Freedman-Lane (1983) nuisance-covariate control (fit distance on
`|length difference|`, permute genre labels against the fixed residual), which for a linear
group-mean-difference statistic is mathematically equivalent to Freedman and Lane's residual-permute
procedure. Three sources are reported side by side per metric: `raw`, `length_controlled`, and
(for every metric except `content_distance` itself) `length_and_content_controlled`, which
additionally residualizes on `content_distance` as a second covariate, isolating a structural
metric's signal from topic.

### Per-genre breakdown

Like the genre benchmark's per-genre extension above, the pooled test only answers whether
genre affects a distance metric at all, not which genres carry that signal.
`trajectory.genre_breakdown.joint_genre_breakdown_permutation_test` restricts the same permutation
test to one genre's one-vs-rest population at a time, reusing `genre.permutation.one_vs_rest_masks`
directly, with a Westfall-Young (1993) maxT correction across the 7 genres, computed once per
(model, metric, source). The population and same-genre sums are reformulated as quadratic forms
over the n x n psalm-pair distance matrix rather than materialized per permutation per pair, since
the naive per-pair form does not scale past a few thousand permutations at 150 psalms.

| script | computes |
|---|---|
| `compute_profiles.py` | content centroid, structural profile, and geometry curves per psalm, every model |
| `validate_against_genre.py` | pooled permutation test of within/between-genre distance (raw, length-controlled, length-and-content-controlled), plus a per-genre one-vs-rest breakdown |
| `export_ui_rows.py` | selects the UI's trajectory columns from `validate_against_genre.py`'s CSVs |

```bash
DATA=/path/to/tehillim-data/benchmark=trajectory/domain=semantic
.venv/bin/python -m trajectory.scripts.compute_profiles \
  /path/to/tehillim-embeddings/data/domain=semantic \
  --output-dir "$DATA/stage=profiles"
.venv/bin/python -m trajectory.scripts.validate_against_genre \
  /path/to/genre-labels.csv \
  "$DATA/stage=profiles/trajectory_distances.parquet" \
  --output "$DATA/stage=raw/validate_against_genre.csv" \
  --breakdown-output "$DATA/stage=raw/validate_against_genre_by_genre.csv"
.venv/bin/python -m trajectory.scripts.export_ui_rows \
  "$DATA/stage=raw/validate_against_genre.csv" \
  --breakdown-csv "$DATA/stage=raw/validate_against_genre_by_genre.csv" \
  --output "$DATA/stage=ui/ui_rows.json" \
  --breakdown-output "$DATA/stage=ui/ui_rows_by_genre.json"
```

## Results UI

The actual results page lives in a separate repo,
[tehillim-ui](https://github.com/rdtaylorjr/tehillim-ui) — a small TypeScript project with no
dependency on this repo's Python code, only on the JSON files this repo produces. `ui_export.export`
selects one domain's UI columns into `ui_<domain>.json`; `ui_export.scripts.build_ui_page` then
injects a tehillim-ui build (its bundle plus `template.html`) and every domain's JSON into one
final, self-contained `ui.html`, filling in an empty six-table placeholder for any domain (e.g.
`phonology`, `discourse`) that has no data yet.

```bash
git clone https://github.com/rdtaylorjr/tehillim-ui /path/to/tehillim-ui
cd /path/to/tehillim-ui && npm install && npm run build && npm run arch-lint && cd -
.venv/bin/python -m ui_export.scripts.build_ui_page \
  /path/to/tehillim-data/ui_semantic.json \
  /path/to/tehillim-data/ui_lexical.json \
  /path/to/tehillim-data/ui_morphology.json \
  /path/to/tehillim-data/ui_syntax.json \
  --template /path/to/tehillim-ui/template.html \
  --bundle /path/to/tehillim-ui/dist/app.bundle.js \
  --output ui.html
```

## Usage

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

Score one embedding file:

```bash
.venv/bin/python -m parallelism.evaluate \
  /path/to/tehillim-embeddings/data/domain=semantic/model=bge_m3/text=vocalized/part-0.parquet
```

Score every model in a `tehillim-embeddings` checkout's data directory:

| script | computes |
|---|---|
| `compare_models.py` | separation AUC, discrimination, type-stratified permutation test, MRR/Recall@k, overall and per-type |
| `compare_true_similarity.py` | calibrated effect size of true-pair similarity, overall and per-type |
| `compare_baseline.py` | true-pair vs. unmarked-bicola Average Precision, AUC, effect sizes |
| `export_detail.py` | row-per-observation raw similarity, rank, and vs-baseline detail (no aggregation) |
| `compute_bootstrap_cis.py` | psalm-clustered BCa 95% CI on AP, gap, and AUC, overall + per-type |
| `build_master_report.py` | joins the CSVs above into one master Parquet set |

```bash
.venv/bin/python -m parallelism.scripts.compare_models \
  /path/to/tehillim-embeddings/data/domain=semantic --output results.csv
```

`model` in every output splits into `model_base` and `text_variant` (consonantal/vocalized/
cantillation) for cross-variant comparison. `dataset_identifier()` derives this from the
Hive-partitioned path (`model=<slug>/text=<variant>/`), not the filename.

### Incremental caching

Every comparison script above (`compare_models.py`, `compare_baseline.py`,
`compare_true_similarity.py`, `compare_calibrated.py`, `compute_bootstrap_cis.py`,
`compare_by_genre.py`, both `export_detail.py` scripts, `compute_profiles.py`) reads its
`--output` path as an implicit cache. A model already present there is skipped on rerun, and its
cached row is kept in the final output (`library.incremental_cache.load_cached_rows` for a single
CSV, `load_cached_parquet_set` for the multi-file detail exports). Delete the output file, or a
row from it, to force that model to rescore. Four genre scripts (`compare_calibrated.py`,
`compute_bootstrap_cis.py`, `export_detail.py`, `compare_by_genre.py`) also skip, rather than
raise, a model whose psalm population is too small for its background similarity distribution to
have any variance, since one already-cached run is not worth losing to a single degenerate model.

## Test

```bash
.venv/bin/pytest && .venv/bin/ruff check src tests && .venv/bin/mypy src
```

## Family

* [tehillim-embeddings](https://github.com/rdtaylorjr/tehillim-embeddings): the embedding vectors
  scored here
* [tehillim-ui](https://github.com/rdtaylorjr/tehillim-ui): the results page that renders this
  repo's `ui_<domain>.json` output
* [tehillim-data](https://github.com/rdtaylorjr/tehillim-data): hosts this repo's Parquet/CSV/JSON
  output
* [bhsa](https://github.com/etcbc/bhsa): the core text and linguistic annotation for the Hebrew
  Bible

## Citations

The parallelism structure and genre classifications used throughout this repo (`parallel_*` BHSA
features and the genre-classification data) are derived from Logos Bible Software's interactive
Psalm resource, used with permission. Chicago 17th ed. format; place and publisher verified
directly against the publisher's own product pages and corporate records:

> Witthoff, David, Kris Lyle, Matt Nerdahl, Jimmy Parks, and Elliot Ritzema. *Psalms Explorer
> Dataset*. Edited by Eli Evans. Bellingham, WA: Logos Bible Software.
> https://www.logos.com/product/54188/psalms-explorer-dataset.

## License

MIT

## Author

* [Russell D. Taylor Jr.](mailto:rdtaylorjr@gatech.edu)
