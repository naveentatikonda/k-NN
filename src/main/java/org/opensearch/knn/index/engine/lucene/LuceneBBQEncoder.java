/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.lucene;

import com.google.common.collect.ImmutableSet;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.Parameter;
import org.opensearch.knn.index.mapper.CompressionLevel;

import java.util.List;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.ENCODER_BBQ;
import static org.opensearch.knn.common.KNNConstants.LUCENE_BBQ_BITS;
import static org.opensearch.knn.common.KNNConstants.LUCENE_BBQ_DEFAULT_BITS;

/**
 * Lucene BBQ (Better Binary Quantization) encoder
 */
public class LuceneBBQEncoder implements Encoder {
    private static final Set<VectorDataType> SUPPORTED_DATA_TYPES = ImmutableSet.of(VectorDataType.FLOAT);

    private final static List<Integer> LUCENE_BBQ_BITS_SUPPORTED = List.of(1);

    private final static MethodComponent METHOD_COMPONENT = MethodComponent.Builder.builder(ENCODER_BBQ)
        .addSupportedDataTypes(SUPPORTED_DATA_TYPES)
        .addParameter(
            LUCENE_BBQ_BITS,
            new Parameter.IntegerParameter(LUCENE_BBQ_BITS, LUCENE_BBQ_DEFAULT_BITS, (v, context) -> LUCENE_BBQ_BITS_SUPPORTED.contains(v))
        )
        .build();

    @Override
    public MethodComponent getMethodComponent() {
        return METHOD_COMPONENT;
    }

    @Override
    public CompressionLevel calculateCompressionLevel(
        MethodComponentContext methodComponentContext,
        KNNMethodConfigContext knnMethodConfigContext
    ) {
        return CompressionLevel.x32;
    }
}
