/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.Version;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

/**
 * Figures out what {@link KNNEngine} to use based on configuration details
 */
public final class EngineResolver {

    public static final EngineResolver INSTANCE = new EngineResolver();

    private EngineResolver() {}

    /**
     * Based on the provided {@link Mode} and {@link CompressionLevel}, resolve to a {@link KNNEngine}.
     *
     * @param knnMethodConfigContext configuration context
     * @param knnMethodContext KNNMethodContext
     * @param requiresTraining whether config requires training
     * @return {@link KNNEngine}
     */
    public KNNEngine resolveEngine(
        KNNMethodConfigContext knnMethodConfigContext,
        KNNMethodContext knnMethodContext,
        boolean requiresTraining
    ) {
        // User configuration gets precedence
        if (knnMethodContext != null && knnMethodContext.isEngineConfigured()) {
            return knnMethodContext.getKnnEngine();
        }

        // Faiss is the only engine that supports training, so we default to faiss here for now
        if (requiresTraining) {
            return KNNEngine.FAISS;
        }

        Mode mode = knnMethodConfigContext.getMode();
        CompressionLevel compressionLevel = knnMethodConfigContext.getCompressionLevel();
        // If both mode and compression are not specified, we can just default
        if (Mode.isConfigured(mode) == false && CompressionLevel.isConfigured(compressionLevel) == false) {
            return KNNEngine.DEFAULT;
        }

        // For 1x, we need to default to faiss if mode is provided and use nmslib otherwise
        if (CompressionLevel.isConfigured(compressionLevel) == false || compressionLevel == CompressionLevel.x1) {
            return mode == Mode.ON_DISK ? KNNEngine.FAISS : KNNEngine.NMSLIB;
        }

        // 4x is supported by Lucene engine before version 2.19.0
        if (knnMethodConfigContext.getVersionCreated().before(Version.V_2_19_0) && compressionLevel == CompressionLevel.x4) {
            return KNNEngine.LUCENE;
        }

        if (compressionLevel == CompressionLevel.x4) {
            return KNNEngine.FAISS;
        }

        return KNNEngine.FAISS;
    }
}
