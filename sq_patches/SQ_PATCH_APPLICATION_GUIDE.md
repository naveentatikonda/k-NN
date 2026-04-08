# SQ Backport Patch Application Guide

This guide explains how to apply the Scalar Quantization (SQ) backport patches onto the `3.3` branch of the OpenSearch k-NN plugin. The backport is split into 3 sequential patches that must be applied in order.

## Patches Overview

| Patch File | Description | Commits | Base Branch |
|---|---|---|---|
| `phase1_3.4_prereqs.patch` | Prerequisite commits from 3.4 (MOS warmup, query fixes, race condition fix) | 7 | `3.3` tip (`d8999306`) |
| `phase2_3.5_prereqs.patch` | Prerequisite commits from 3.5 (scorer fixes, SIMD V2, nested docs, build tasks) | 26 | After Phase 1 |
| `phase3_3.6_sq_and_fixes.patch` | SQ feature from 3.6 + Lucene 10.3 compatibility fixes + KNN1040→KNN1030 codec refactoring | 53 | After Phase 2 |

Total: **86 commits**

---

## Applying the Patches

### Step 1: Start from the 3.3 branch tip

```bash
cd /path/to/k-NN
git checkout 3.3
git checkout -b 3.3-sq
```

Verify you are at the correct starting point:

```bash
git log --oneline -1
# Expected: d8999306 Fix quant bwc main (#2994) (#2997)
```

### Step 2: Apply Phase 1 — 3.4 Prerequisites

```bash
git am phase1_3.4_prereqs.patch
```

Verify:

```bash
git log --oneline -7
```

Expected commits (newest first):
```
7e4a36ed Include opensearchknn_simd in build configurations (#3025) (#3028)
19d30af3 Ci runner macos fix and faiss distance fix backport to 3.4 (#3018)
97864901 Change to use the function defined in core to check if system generate search factory is enabled or not (#2927)
80840db3 Fix race condition on transforming vector in KNNQueryBuilder. (#2974)
8ec31ea6 [Bug Fix] Fix NativeEngineKNNQuery to return correct results as 'totalHits' when memory optimized search is enabled. (#2965)
08368938 Added functionality for warmup with memory optimized search (#2954) (#2954)
e96178bd Removed VectorSearchHolders map from NativeEngines990KnnVectorsReader to improve heap utilization (#2948)
```

### Step 3: Apply Phase 2 — 3.5 Prerequisites

```bash
git am phase2_3.5_prereqs.patch
```

Verify:

```bash
git log --oneline -26
```

### Step 4: Apply Phase 3 — 3.6 SQ Feature + Fixes

```bash
git am phase3_3.6_sq_and_fixes.patch
```

Verify:

```bash
git log --oneline -3
```

Expected top commits:
```
e37c2cd7 Refactor KNNCodec
502b214b Fix package name and OpenSearch version in tests
e4d5cf2c Fix Lucene package name and OpenSearch version errors
```

---

## Handling Failures

If `git am` fails on any patch:

### Option A: Resolve and continue

```bash
# See which file(s) have conflicts
git status

# Resolve conflicts in the affected files, then:
git add <resolved-files>
git am --continue
```

### Option B: Skip a problematic commit

```bash
git am --skip
```

### Option C: Abort and start over

```bash
git am --abort
```

### Option D: Apply with 3-way merge (often resolves simple conflicts automatically)

```bash
git am --3way phase1_3.4_prereqs.patch
```

---

## Verification

### Compile Check

```bash
./gradlew compileJava
./gradlew compileTestJava
```

Both should report `BUILD SUCCESSFUL`.

### Unit Tests

```bash
./gradlew test
```

### Integration Tests

Run SQ-specific ITs against the custom OpenSearch distribution:

```bash
DIST="/path/to/opensearch-min-3.3.2-SNAPSHOT-darwin-arm64.tar.gz"

# Faiss SQ ITs
./gradlew :integTest -PcustomDistributionUrl="$DIST" -Dtests.class="org.opensearch.knn.index.FaissSQIT"
./gradlew :integTest -PcustomDistributionUrl="$DIST" -Dtests.class="org.opensearch.knn.index.FaissSQMappingIT"
./gradlew :integTest -PcustomDistributionUrl="$DIST" -Dtests.class="org.opensearch.knn.index.MOSFaissSQIndexIT"

# Lucene SQ ITs
./gradlew :integTest -PcustomDistributionUrl="$DIST" -Dtests.class="org.opensearch.knn.index.LuceneSQFlatIT"
./gradlew :integTest -PcustomDistributionUrl="$DIST" -Dtests.class="org.opensearch.knn.index.LuceneSQMappingIT"
```

---

## Key Notes

- **Lucene version**: The 3.3 branch uses Lucene 10.3. All `lucene104` package references and `Lucene104*` class names from 3.6 have been replaced with `lucene103` / `Lucene103*` in the Phase 3 patch.
- **Codec naming**: The codec has been renamed from `KNN1040Codec` (3.6) to `KNN1030Codec` (3.3). All internal SQ classes (`Faiss1040*`, `KNN1040ScalarQuantized*`, etc.) have also been renamed to `1030`.
- **Derived source commits**: 5 derived source commits were intentionally excluded — they depend on `SourceFieldMapper.getIncludes()`/`getExcludes()` which are not available in OpenSearch 3.3 core, and are not needed for the SQ feature.
- **Patch format**: Patches were generated with `git format-patch --stdout`, preserving full commit metadata (author, date, message). `git am` will replay them as individual commits.
