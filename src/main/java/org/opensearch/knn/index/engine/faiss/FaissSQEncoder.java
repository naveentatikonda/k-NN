/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import com.google.common.collect.ImmutableSet;
import org.opensearch.common.ValidationException;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.Parameter;
import org.opensearch.knn.index.mapper.CompressionLevel;

import java.util.Objects;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_CLIP;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_ENCODER_FP16;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_ENCODER_TYPES;
import static org.opensearch.knn.common.KNNConstants.FAISS_SQ_TYPE;

/**
 * Faiss SQ encoder
 */
public class FaissSQEncoder implements Encoder {

    private static final Set<VectorDataType> SUPPORTED_DATA_TYPES = ImmutableSet.of(VectorDataType.FLOAT);

    private final static MethodComponent METHOD_COMPONENT = MethodComponent.Builder.builder(ENCODER_SQ)
        .addSupportedDataTypes(SUPPORTED_DATA_TYPES)
        .addParameter(
            FAISS_SQ_TYPE,
            new Parameter.StringParameter(FAISS_SQ_TYPE, FAISS_SQ_ENCODER_FP16, (v, context) -> FAISS_SQ_ENCODER_TYPES.contains(v))
        )
        .addParameter(FAISS_SQ_CLIP, new Parameter.BooleanParameter(FAISS_SQ_CLIP, false, (v, context) -> Objects.nonNull(v)))
        .setKnnLibraryIndexingContextGenerator(
            ((methodComponent, methodComponentContext, knnMethodConfigContext) -> MethodAsMapBuilder.builder(
                FAISS_SQ_DESCRIPTION,
                methodComponent,
                methodComponentContext,
                knnMethodConfigContext
            ).addParameter(FAISS_SQ_TYPE, "", "").build())
        )
        // .setKnnLibraryIndexingContextGenerator(((methodComponent, methodComponentContext, knnMethodConfigContext) -> {
        // QuantizationConfig quantizationConfig = QuantizationConfig.EMPTY;
        // String sqEncoderType = (String) methodComponentContext.getParameters().getOrDefault(FAISS_SQ_TYPE, FAISS_SQ_ENCODER_FP16);
        // if(FAISS_SQ_ENCODER_FP16.equals(sqEncoderType)) {
        // return MethodAsMapBuilder.builder(
        // FAISS_SQ_DESCRIPTION,
        // methodComponent,
        // methodComponentContext,
        // knnMethodConfigContext
        // ).addParameter(FAISS_SQ_TYPE, "", "").build();
        //
        // }
        // else if (FAISS_SQ_ENCODER_INT8.equals(sqEncoderType)) {
        // quantizationConfig = QuantizationConfig.builder().quantizationType(ScalarQuantizationType.EIGHT_BIT).build();
        // }
        // return KNNLibraryIndexingContextImpl.builder().quantizationConfig(quantizationConfig).parameters(new HashMap<>() {
        // {
        // put(INDEX_DESCRIPTION_PARAMETER, FAISS_FLAT_DESCRIPTION);
        // }
        // }).build();
        // }))
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
        if (methodComponentContext.getParameters().containsKey(FAISS_SQ_TYPE) == false) {
            return CompressionLevel.NOT_CONFIGURED;
        }

        // Map the number of bits passed in, back to the compression level
        Object value = methodComponentContext.getParameters().get(FAISS_SQ_TYPE);
        ValidationException validationException = METHOD_COMPONENT.getParameters()
            .get(FAISS_SQ_TYPE)
            .validate(value, knnMethodConfigContext);
        if (validationException != null) {
            throw validationException;
        }

        String SQEncoderType = (String) value;
        if (FAISS_SQ_ENCODER_FP16.equals(SQEncoderType)) {
            return CompressionLevel.x2;
        }

        return CompressionLevel.x4;
    }
}
