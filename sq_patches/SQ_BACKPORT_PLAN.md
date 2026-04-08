# SQ Feature Backport Plan: 3.6 → 3.3-sq

## Goal
Cherry-pick all SQ (Scalar Quantization) related commits from `3.6` branch (after commit `89acc374`) onto the `3.3-sq` branch. Before that, identify and cherry-pick prerequisite commits from `3.4` and `3.5` branches that touch the same files modified by the SQ feature.

## Current State
- **3.3-sq branch tip**: `d8999306` (same as `3.3` branch tip)
- **SQ commits start after**: `89acc374` (Increment version to 3.6.0-SNAPSHOT) on `3.6`
- **Total commits on 3.6 after starting point**: 55
- **Already in 3.3**: PRs #2963 (FP16 native scoring), #2933 (optimistic search), #2994 (quant bwc fix), #2925 (delegate methods), #2921 (nested exact search rescore)

---

## Step 1: Cherry-pick Prerequisite Commits from 3.4

These are commits in `3.4` (not in `3.3`) that touch files also modified by the SQ feature on `3.6`. Listed in **chronological order** (oldest first).

> **Already in 3.3 — SKIP**: `a5966d7a` (#2921), `0accbfc4` (#2925), `a95a7d0b` (#2933), `575e66da` (#2963), `fd71813f` (#2994)

| # | Commit | Description | Key Overlapping Files | Status |
|---|--------|-------------|----------------------|--------|
| 1 | ~~`56c1cd63`~~ | ~~Do not apply memory optimized search for old indices (#2918)~~ | ~~`MemoryOptimizedSearchSupportSpec`, `EngineFieldMapper`, `KNNVectorFieldType`, `KNNVectorFieldMapperTests`~~ | ⏭️ **SKIPPED** — already in 3.3 as `736808fc` (#2949) |
| 2 | `d05de18c` | Removed VectorSearchHolders map from NativeEngines990KnnVectorsReader (#2948) | `NativeEngines990KnnVectorsReader`, `NativeEngines990KnnVectorsReaderTests` | ✅ Cherry-picked |
| 3 | ~~`47562b79`~~ | ~~Disable cgroup detection in IT (#2952)~~ | ~~`qa/restart-upgrade/build.gradle`~~ | ⏭️ **SKIPPED** — became empty, changes already present in 3.3 |
| 4 | `5e9e2dcc` | Added functionality for warmup with memory optimized search (#2954) | `KNNIndexShard`, `MemoryOptimizedSearchWarmup`, `MemoryOptimizedSearchWarmupTests` | ✅ Cherry-picked |
| 5 | `5388ea31` | Fix NativeEngineKNNQuery to return correct totalHits with MOS (#2965) | `NativeEngineKnnVectorQuery`, `KNNTestCase`, `OptimisticSearchTests` | ✅ Cherry-picked |
| 6 | `32b84fb8` | Refactor to not use parallel for MMR rerank (#2968) | _(minor overlap, may be skippable — verify)_ | ⏭️ Not cherry-picked (no file overlap with SQ) |
| 7 | `51b1c8e6` | Fix race condition on transforming vector in KNNQueryBuilder (#2974) | `KNNVectorFieldType`, `VectorTransformerFactory`, `KNNQueryBuilder`, `KNNQueryFactory`, `KNNPlugin` | ✅ Cherry-picked |
| 8 | ~~`cb572c12`~~ | ~~Gradle 9.2.0 and GitHub Actions JDK 25 Upgrade (#2984)~~ | ~~`build.gradle`~~ | ⏭️ **SKIPPED** per plan |
| — | `ae67de40` | Change to use the function defined in core to check if system generate search factory is enabled (#2927) | `KNNPlugin` | ✅ Cherry-picked |
| 9 | `ecb0df5d` | CI runner macos fix and faiss distance fix (#3018) | `KNNQueryBuilderTests` | ✅ Cherry-picked |
| 10 | `88e36980` | Include opensearchknn_simd in build configurations (#3025) (#3028) | _(build config)_ | ✅ Cherry-picked |

### Notes on 3.4 commits
- Commits #1-5 and #7 are **high priority** — they modify core files (searcher, query, mapper, warmup) that the SQ feature directly depends on.
- Commits #6, #8, #9, #10 are **lower priority** — they touch build/test files. Evaluate if they cause merge conflicts; if not, they can be skipped or cherry-picked as needed.

---

## Step 2: Cherry-pick Prerequisite Commits from 3.5

These are commits in `3.5` (not in `3.4`) that touch files also modified by the SQ feature. Listed in **chronological order** (oldest first).

| # | Commit | Description | Key Overlapping Files | Status |
|---|--------|-------------|----------------------|--------|
| 1 | `3343b76c` | Fixed Faiss score to distance calculation (#2992) | `KNNQueryBuilderTests` | ✅ Cherry-picked |
| 2 | `d2e228d0` | Block index creation for byte-cosinesimilarity-faiss indices (#3002) | `KNNVectorFieldMapperTests` | ✅ Cherry-picked |
| 3 | `2be906c6` | Fix indexing for 16x and 8x compression (#3019) | `KNNConstants`, `ModeAndCompressionIT` | ✅ Cherry-picked |
| 4 | ~~`515957ba`~~ | ~~Include opensearchknn_simd in build configurations (#3025)~~ | ~~_(build config)_~~ | ⏭️ **SKIPPED** — empty, already present from Phase 1 (`88e36980`) |
| 5 | `1a123af9` | Added library loading class with synchronized behavior (#3024) | `build.gradle`, `FaissService`, `SimdVectorComputeService` | ✅ Cherry-picked |
| 6 | `01d42bf7` | Index setting to disable exact search after ANN with Faiss filters (#3022) | `KNNSettings` | ✅ Cherry-picked |
| 7 | `21ffd2d1` | Create build graph gradle task (#3032) | `build.gradle` | ✅ Cherry-picked |
| 8 | `527d6e69` | Added gradle task validateLibraryUsage (#3033) | `build.gradle` | ✅ Cherry-picked |
| 9 | ~~`5ff7253f`~~ | ~~Regex for derived source support (#3031)~~ | ~~`KNN10010DerivedSourceStoredFieldsReader`, `DerivedSourceIT`~~ | ❌ **REMOVED** — dropped via rebase (derived source not needed for SQ, causes issues on 3.3) |
| 10 | `a799b258` | Fixes the build (#3050) | `build.gradle` | ✅ Cherry-picked |
| 11 | `ee1eb256` | Correct ef_search parameter for Lucene engine and reduce to top K (#3037) | `KNNQueryFactory`, `InternalNestedKnnFloatVectorQuery`, `NestedKnnVectorQueryFactory`, `OSDiversifyingChildrenFloatKnnVectorQuery`, `OSKnnFloatVectorQuery`, `LuceneEngineIT` | ✅ Cherry-picked |
| 12 | `946e038e` | Join filter clauses of nested k-NN queries to root-parent scope (#2990) | `KNNQueryFactoryTests`, `ExpandNestedDocsIT` | ✅ Cherry-picked |
| 13 | `d6f8822a` | Changed warmup seek to use long instead of int to avoid overflow (#3067) | `MemoryOptimizedSearchWarmup` | ✅ Cherry-picked |
| 14 | `2366fc32` | Fix nested docs and exact search query when some docs has no vector field (#3051) | `ExactSearcher`, `ExactKNNIterator`, all `*ExactKNNIterator` classes, `NativeEngineKnnVectorQuery`, `TestVectorValues`, `ExpandNestedDocsIT` | ✅ Cherry-picked |
| 15 | `16662c3f` | Fix memory optimized weight to use correct `k` when rescoring (#3061) | `NativeEngineKnnVectorQuery`, `OptimisticSearchTests` | ✅ Cherry-picked (manual conflict resolution in `OptimisticSearchTests.java`) |
| 16 | `11844328` | Fix MOS reentrant search bug in byte index (#3071) | `NativeEngineKnnVectorQuery`, `AbstractMemoryOptimizedKnnSearchIT`, `OptimisticSearchTests`, `Documents` | ✅ Cherry-picked (manual conflict resolution in `OptimisticSearchTests.java`) |
| 17 | `c7a48381` | Fix BWC tests which haven't been run (noop) (#3068) | `qa/restart-upgrade/build.gradle`, `qa/rolling-upgrade/build.gradle` | ✅ Cherry-picked |
| 18 | `29420a7d` | Added new exception type to signify expected warmup behavior (#3070) | `NativeEngines990KnnVectorsReader`, `MemoryOptimizedSearchWarmup`, `FaissMemoryOptimizedSearcher`, warmup tests | ✅ Cherry-picked |
| 19 | `7d30ea5e` | Bulk SIMD V2 Implementation (#3075) | `arm_neon_simd_similarity_function.cpp`, `avx512_simd_similarity_function.cpp`, `default_simd_similarity_function.cpp`, `simd_similarity_function_common.cpp` | ✅ Cherry-picked |
| 20 | `df16532a` | Add IT and bwc test with indices containing both vector and non-vector docs (#3064) | `qa/restart-upgrade/build.gradle`, `IndexingIT`, `IndexIT`, `ModeAndCompressionIT`, `KNNRestTestCase` | ✅ Cherry-picked |
| 21 | ~~`efb39942`~~ | ~~Field exclusion in source indexing handling (#3049)~~ | ~~`DerivedSourceIT`~~ | ❌ **REMOVED** — dropped via rebase (`SourceFieldMapper.getIncludes()`/`getExcludes()` not available in 3.3 core) |
| 22 | ~~`b9947885`~~ | ~~Include AdditionalCodecs argument to allow additional Codec registration (#3085)~~ | ~~`KNNCodecService`~~ | ⏭️ **SKIPPED** per plan |
| 23 | `d1fd77ee` | Fix patch to have a valid score conversion for BinaryCagra (#2983) | _(JNI patch file)_ | ✅ Cherry-picked |
| 24 | `46fd2273` | Skipping deletion of .tasks index (#3090) | `KNNConstants` | ✅ Cherry-picked |
| 25 | `1aced12b` | Increase timeout to 40min to FaissIT (#3097) | `FaissIT` | ✅ Cherry-picked |
| 26 | `899e337d` | Backport 2529-to-3.5 (Make Merge in nativeEngine can Abort) (#3111) | `build.gradle`, `init-faiss.cmake`, `faiss_index_service.h`, `FaissService.java`, `DefaultIndexBuildStrategy`, `MemOptimizedNativeIndexBuildStrategy`, `NativeIndexWriter`, `RelocationIT` | ✅ Cherry-picked |
| 27 | `f26290b4` | Fix score conversion logic for radial exact search (#3110) | `ExactSearcher`, `FaissIT` | ✅ Cherry-picked |
| 28 | ~~`072ed4cf`~~ | ~~Simplify DerivedSourceReaders lifecycle (#3139)~~ | ~~`KNN10010DerivedSourceStoredFieldsReader`, `DerivedSourceReaders`, `DerivedSourceReadersTests`~~ | ❌ **REMOVED** — dropped via rebase (derived source not needed for SQ) |
| 29 | `62dc8d5a` | Reenable reindex ITs (#3056) | `KNNRestTestCase` | ✅ Cherry-picked |
| 30 | `52a8bc50` | fix: Use pypi for codecov to address glibc error (#3057) | _(CI config)_ | ✅ Cherry-picked |
| 31 | `7e9ea355` | Update validation for cases when k > total results (#3038) | _(test only)_ | ✅ Cherry-picked |

### Notes on 3.5 commits
- **High priority** (core SQ dependencies): #3, #11, #13, #14, #15, #17, #18, #19, #22, #26, #27
- **Medium priority** (codec/build/infra changes that may cause conflicts): #5, #6, #9, #20, #28
- **Lower priority** (test-only or CI changes): #1, #2, #4, #7, #8, #10, #12, #16, #21, #23, #24, #25, #29, #30, #31

---

## Step 3: Cherry-pick SQ Feature Commits from 3.6

After all prerequisites are in place, cherry-pick the SQ-specific commits from `3.6` (after `89acc374`). These are the 55 commits between `89acc374..3.6`. Key SQ-specific ones:

| Commit | Description |
|--------|-------------|
| `959512db` | Add support for pre-quantized vector exact search (#3095) |
| `0d05a101` | Use pre-quantized vectors for ADC (#3113) |
| `a1946158` | Support lucene bbq flat for 1 bit (#3154) |
| `c9b05881` | Faiss Scalar Quantization Support for 1 bit using MOS (#3208) |
| `f1e822c4` | Bug fix for clip parameter, IT coverage for Faiss SQ 1 bit (#3206) |
| `ab760649` | 1 bit compression support for the Lucene Scalar Quantizer (#3144) |
| `29f86da4` | Fix bug when taking dimension < 8 for Faiss SQ (#3220) |
| `565f7ebc` | Fix bug when dimension <= 56 in Faiss SQ (#3229) |
| `daa8b80b` | Fix naming from BBQ to SQ and add more ITs (#3231) |
| `99c7977a` | Fix Scorer, and max dimension for Lucene SQ 1 bit (#3203) |
| `36bf6fe5` | Add BwC tests for Faiss SQ 1 bit (#3223) |
| `65b749f3` | Fix default encoder to SQ 1 bit for faiss 32x compression (#3210) |
| `b6a5b390` | Added more ITs for LuceneSQ, updated Lucene SQ Encoder behavior (#3219) |
| `6062459d` | Block radial search on Faiss SQ (#3257) |
| `d33f27f8` | Block radial search on Lucene 32x compression for flat and SQ (#3259) |

Plus supporting/dependency commits on 3.6 (scorer refactors, prefetch, codec changes, etc.):

| Commit | Description |
|--------|-------------|
| `e327675a` | Fix score conversion logic for radial exact search (#3110) |
| `20295a0a` | Use right Vector Scorer when segments are initialized using SPI (#3117) |
| `159e40de` | Fix KNN build and run due to Lucene upgrade 10.4.0 (#3135) |
| `7f19fa7d` | KNN1030Codec does not properly support delegation for non-default codec(s) (#3093) |
| `a2372a31` | Fix lucene reduce to topK when rescoring is enabled (#3124) |
| `6c4d4277` | Add NestedBestChildVectorScorer and KnnBinaryDocValuesScorer (#3179) |
| `d7b5ecbd` | Add Prefetch functionality for MOS (#3173) |
| `afa7c5c7` | Introduce NativeEngines990KnnVectorsScorer (#3184) |
| `88cab7c8` | Moved creation of Scorer out of FaissMemoryOptimizedSearcher (#3187) |
| `0ab8e378` | Introduce VectorScorers (#3183) |
| `bd7ebdd5` | Add FaissScorableByteVectorValues (#3192) |
| `ad5684af` | Integrated Prefetch with Fp16 based index for MOS (#3195) |
| `9f626c7e` | Refactor ExactSearcher to use VectorScorer (#3207) |
| `6f00683a` | Add Hamming distance scorer for byte vectors (#3214) |
| `396c7968` | Fix the scorers to use Prefetch scorer (#3248) |
| Others | Remaining infra/bug-fix commits as needed |

---

## Recommended Execution Order

### Phase 1: 3.4 Prerequisites (cherry-pick onto 3.3-sq) — ✅ COMPLETED
```bash
git checkout 3.3-sq

# git cherry-pick 56c1cd63  # MOS for old indices — SKIPPED (already in 3.3 as 736808fc #2949)
git cherry-pick d05de18c  # VectorSearchHolders removal ✅
git cherry-pick 5e9e2dcc  # Warmup with MOS ✅
git cherry-pick 5388ea31  # NativeEngineKNNQuery totalHits fix ✅
git cherry-pick 51b1c8e6  # Race condition in KNNQueryBuilder ✅
git cherry-pick ae67de40  # System generate search factory check (#2927) ✅

# Build/CI
# git cherry-pick 47562b79  # Disable cgroup detection — SKIPPED (already present in 3.3)
# git cherry-pick cb572c12  # Gradle 9.2.0 upgrade — SKIPPED per plan
git cherry-pick ecb0df5d  # CI runner macos fix and faiss distance fix ✅
git cherry-pick 88e36980  # opensearchknn_simd build config ✅
```

### Phase 2: 3.5 Prerequisites (cherry-pick onto 3.3-sq) — ✅ COMPLETED
```bash
git cherry-pick 3343b76c  # Faiss score to distance fix ✅
git cherry-pick d2e228d0  # Block byte-cosinesimilarity-faiss indices ✅
git cherry-pick 2be906c6  # Fix indexing for 16x and 8x compression ✅
# git cherry-pick 515957ba  # opensearchknn_simd build config — SKIPPED (already present from Phase 1)
git cherry-pick 1a123af9  # Library loading class ✅
git cherry-pick 01d42bf7  # Index setting to disable exact search ✅
git cherry-pick 21ffd2d1  # Create build graph gradle task ✅
git cherry-pick 527d6e69  # Added gradle task validateLibraryUsage ✅
# git cherry-pick 5ff7253f  # Regex for derived source support — REMOVED (derived source not needed for SQ)
git cherry-pick a799b258  # Fixes the build ✅
git cherry-pick ee1eb256  # Correct ef_search for Lucene ✅
git cherry-pick 946e038e  # Join filter clauses nested queries ✅
git cherry-pick d6f8822a  # Warmup seek overflow fix ✅
git cherry-pick 2366fc32  # Fix nested docs and exact search ✅
git cherry-pick 16662c3f  # Fix MOS k value with rescoring ✅ (manual conflict in OptimisticSearchTests.java)
git cherry-pick 11844328  # MOS reentrant search bug ✅ (manual conflict in OptimisticSearchTests.java)
git cherry-pick c7a48381  # Fix BWC tests noop ✅
git cherry-pick 29420a7d  # Expected warmup behavior exception ✅
git cherry-pick 7d30ea5e  # Bulk SIMD V2 ✅
git cherry-pick df16532a  # IT and bwc test with vector/non-vector docs ✅
# git cherry-pick efb39942  # Field exclusion in source indexing handling — REMOVED (compilation error on 3.3)
# git cherry-pick b9947885  # AdditionalCodecs argument — SKIPPED per plan
git cherry-pick d1fd77ee  # BinaryCagra score conversion patch ✅
git cherry-pick 46fd2273  # Skipping .tasks index deletion ✅
git cherry-pick 1aced12b  # Increase timeout to 40min to FaissIT ✅
git cherry-pick 899e337d  # Make Merge in nativeEngine can Abort ✅
git cherry-pick f26290b4  # Fix score conversion for radial exact search ✅
# git cherry-pick 072ed4cf  # Simplify DerivedSourceReaders lifecycle — REMOVED (derived source not needed for SQ)
git cherry-pick 62dc8d5a  # Reenable reindex ITs ✅
git cherry-pick 52a8bc50  # fix: Use pypi for codecov ✅
git cherry-pick 7e9ea355  # Update validation for k > total results ✅
```

### Phase 3: 3.6 SQ Feature Commits (cherry-pick onto 3.3-sq) — ✅ COMPLETED
Cherry-picked 53 commits from `89acc374..3.6` (excluding release notes/changelog). 1 manual conflict resolution (`DerivedSourceReaders.java` at commit 25/53). 2 derived source commits from 3.6 were also dropped during the post-Phase-3 rebase.

Commits dropped from 3.6 during rebase:
- `5c061be9` — Derived source dynamic template fix (#3035)
- `58f38958` — Simplify DerivedSourceReaders lifecycle by removing manual ref-counting (#3138)

---

## Post Cherry-pick Compilation Errors & Resolution

After completing Phase 1, 2, and 3, `./gradlew compileJava` found **2 errors**:

| File | Line | Error | Introduced By |
|------|------|-------|---------------|
| `src/main/java/org/opensearch/knn/index/util/IndexUtil.java` | 507 | `cannot find symbol: method getIncludes()` on `SourceFieldMapper` | `efb39942` — Field exclusion in source indexing handling (#3049) |
| `src/main/java/org/opensearch/knn/index/util/IndexUtil.java` | 508 | `cannot find symbol: method getExcludes()` on `SourceFieldMapper` | `efb39942` — Field exclusion in source indexing handling (#3049) |

**Root cause**: `SourceFieldMapper.getIncludes()` and `getExcludes()` were added in OpenSearch core after 3.3.

**Resolution**: Dropped all 5 derived source / field exclusion commits via `git rebase --onto` since they are not needed for the SQ feature:

| Commit | Description | Phase |
|--------|-------------|-------|
| `e9dda014` | Regex for derived source support (#3031) | Phase 2 #9 |
| `5b579f06` | Field exclusion in source indexing handling (#3049) | Phase 2 #21 |
| `ae8d0b73` | Simplify DerivedSourceReaders lifecycle (#3139) | Phase 2 #28 |
| `5c061be9` | Derived source dynamic template fix (#3035) | Phase 3 |
| `58f38958` | Simplify DerivedSourceReaders lifecycle (#3138) | Phase 3 |

Total commits after rebase: **83** (was 88 before dropping 5).

---

## Conflict Resolution Strategy
1. **Resolve conflicts file-by-file** — the 3.3 codebase may have different class structures (e.g., no KNN1040Codec, different Lucene version)
2. **Lucene version differences** — 3.3 uses an older Lucene; codec classes (KNN1040Codec, etc.) may need to be adapted or created fresh
3. **Build file conflicts** — `build.gradle` will have version differences; keep 3.3 version numbers but adopt structural changes
4. **Test conflicts** — test files may reference classes/methods not yet present; resolve after all source changes are in place

## Validation
After all cherry-picks:
1. `./gradlew build` — ensure compilation succeeds
2. `./gradlew test` — run unit tests
3. `./gradlew integTest` — run integration tests
4. Specifically validate SQ-related ITs: `FaissSQIT`, `LuceneSQFlatIT`, `FaissSQMappingIT`, `LuceneSQMappingIT`, `MOSFaissSQIndexIT`
