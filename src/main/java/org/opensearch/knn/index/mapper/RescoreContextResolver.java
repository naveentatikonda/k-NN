/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import lombok.Builder;
import lombok.Getter;
import org.opensearch.Version;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.rescore.RescoreContext;

/**
 * Resolves the default {@link RescoreContext} for a given compression level and index configuration.
 *
 * <p>Priority-ordered rules — first match wins. Each rule is a pair of static methods:
 * {@code isX(request)} declares the condition, and {@code xContext()} produces the {@link RescoreContext}
 * to return when the condition matches. To add a new rule (for example, a future {@code bits=8} config),
 * add one predicate method plus one context factory, and insert an {@code if (predicate) return context;}
 * line at the correct priority in {@link #resolve(Request)}.
 *
 * <p>Rule priority (high to low):
 * <ol>
 *   <li>Multi-bit SQ at x32 — fixed factor, override disabled.</li>
 *   <li>Multi-bit SQ at x16 or x8 — factor 1, override allowed.</li>
 *   <li>Flat method at x32 — flat oversample factor.</li>
 *   <li>Compression level not in a rescore-eligible mode — return {@code null}.</li>
 *   <li>Lucene scalar quantizer at x32 on 3.6.0+ — Lucene-specific factor.</li>
 *   <li>Legacy x4 on pre-3.1.0 indices — return {@code null}.</li>
 *   <li>Non-x4 with dimension &le; threshold — boosted factor for small dims.</li>
 *   <li>Fallback — the compression level's own default context.</li>
 * </ol>
 */
public final class RescoreContextResolver {

    private static final float FLAT_OVERSAMPLE_FACTOR = 2.0f;
    private static final float SQ_MULTI_BIT_DEFAULT_OVERSAMPLE_FACTOR = 1.0f;

    private RescoreContextResolver() {}

    /**
     * Inputs to a rescore-context resolution. Immutable; use {@link #builder()} to construct.
     */
    @Getter
    @Builder
    public static final class Request {
        private final CompressionLevel compression;
        private final Mode mode;
        private final int dimension;
        private final Version version;
        private final boolean isFlatMethod;
        private final boolean isSQMultiBit;
        private final KNNEngine engine;
    }

    /**
     * Resolves the default {@link RescoreContext} for the given request. Returns {@code null} when
     * no default rescore applies. See the class Javadoc for the rule priority order.
     */
    public static RescoreContext resolve(final Request req) {
        if (isSQMultiBitX32(req)) {
            return sqMultiBitX32Context();
        }
        if (isSQMultiBitX16OrX8(req)) {
            return sqMultiBitDefaultContext();
        }
        if (isFlatMethodAtX32(req)) {
            return flatMethodAtX32Context();
        }
        if (isNotARescoreMode(req)) {
            return null;
        }
        if (isLuceneScalarQuantizerAtX32(req)) {
            return luceneScalarQuantizerAtX32Context();
        }
        if (isLegacyX4PreV310(req)) {
            return null;
        }
        if (needsSmallDimensionBoost(req)) {
            return smallDimensionBoostContext();
        }
        return req.compression.getDefaultRescoreContext();
    }

    // --- Rule 1: Multi-bit SQ at x32 ---

    private static boolean isSQMultiBitX32(final Request req) {
        return req.isSQMultiBit && req.compression == CompressionLevel.x32;
    }

    private static RescoreContext sqMultiBitX32Context() {
        return RescoreContext.builder()
            .oversampleFactor(RescoreContext.FAISS_SCALAR_QUANTIZED_INDEX_OVERSAMPLE_FACTOR)
            .allowOverrideOversampleFactor(false)
            .userProvided(false)
            .build();
    }

    // --- Rule 2: Multi-bit SQ at x16 or x8 ---

    private static boolean isSQMultiBitX16OrX8(final Request req) {
        return req.isSQMultiBit && (req.compression == CompressionLevel.x16 || req.compression == CompressionLevel.x8);
    }

    private static RescoreContext sqMultiBitDefaultContext() {
        return RescoreContext.builder().oversampleFactor(SQ_MULTI_BIT_DEFAULT_OVERSAMPLE_FACTOR).userProvided(false).build();
    }

    // --- Rule 3: Flat method at x32 ---

    private static boolean isFlatMethodAtX32(final Request req) {
        return req.compression == CompressionLevel.x32 && req.isFlatMethod;
    }

    private static RescoreContext flatMethodAtX32Context() {
        return RescoreContext.builder().oversampleFactor(FLAT_OVERSAMPLE_FACTOR).userProvided(false).build();
    }

    // --- Rule 4: Mode is not one this compression level rescores ---

    private static boolean isNotARescoreMode(final Request req) {
        return req.compression.getModesForRescore().contains(req.mode) == false;
    }

    // --- Rule 5: Lucene scalar quantizer at x32 on 3.6.0+ ---

    private static boolean isLuceneScalarQuantizerAtX32(final Request req) {
        return req.compression == CompressionLevel.x32
            && req.engine == KNNEngine.LUCENE
            && req.version != null
            && req.version.onOrAfter(Version.V_3_6_0);
    }

    private static RescoreContext luceneScalarQuantizerAtX32Context() {
        return RescoreContext.builder()
            .oversampleFactor(RescoreContext.OVERSAMPLE_FACTOR_DEFAULT_FOR_LUCENE_SCALAR_QUANTIZER_AFTER_V360)
            .userProvided(false)
            .build();
    }

    // --- Rule 6: Legacy x4 on pre-3.1.0 indices — no rescore context ---

    private static boolean isLegacyX4PreV310(final Request req) {
        return req.compression == CompressionLevel.x4 && req.version != null && req.version.before(Version.V_3_1_0);
    }

    // --- Rule 7: Small-dimension boost (non-x4) ---

    private static boolean needsSmallDimensionBoost(final Request req) {
        return req.compression != CompressionLevel.x4 && req.dimension <= RescoreContext.DIMENSION_THRESHOLD;
    }

    private static RescoreContext smallDimensionBoostContext() {
        return RescoreContext.builder().oversampleFactor(RescoreContext.OVERSAMPLE_FACTOR_BELOW_DIMENSION_THRESHOLD).userProvided(false).build();
    }
}
