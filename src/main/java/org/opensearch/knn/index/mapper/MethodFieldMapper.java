/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.apache.lucene.document.FieldType;
import org.opensearch.common.Explicit;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.index.mapper.MapperParsingException;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.index.MethodComponentContext;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.util.KNNEngine;

import java.io.IOException;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.ENCODER_FLAT;
import static org.opensearch.knn.common.KNNConstants.INDEX_DESCRIPTION_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;

/**
 * Field mapper for method definition in mapping
 */
public class MethodFieldMapper extends KNNVectorFieldMapper {

    MethodFieldMapper(
        String simpleName,
        KNNVectorFieldType mappedFieldType,
        MultiFields multiFields,
        CopyTo copyTo,
        Explicit<Boolean> ignoreMalformed,
        boolean stored,
        boolean hasDocValues,
        KNNMethodContext knnMethodContext,
        VectorDataType vectorDataType
    ) {

        super(
            simpleName,
            mappedFieldType,
            multiFields,
            copyTo,
            ignoreMalformed,
            stored,
            hasDocValues,
            knnMethodContext.getMethodComponentContext().getIndexVersion()
        );

        this.knnMethod = knnMethodContext;

        this.fieldType = new FieldType(KNNVectorFieldMapper.Defaults.FIELD_TYPE);

        this.fieldType.putAttribute(DIMENSION, String.valueOf(dimension));
        this.fieldType.putAttribute(SPACE_TYPE, knnMethodContext.getSpaceType().getValue());
        this.fieldType.putAttribute(VECTOR_DATA_TYPE_FIELD, vectorDataType.getValue());

        KNNEngine knnEngine = knnMethodContext.getKnnEngine();
        this.fieldType.putAttribute(KNN_ENGINE, knnEngine.getName());
        if (vectorDataType.equals(VectorDataType.BYTE)) {
            if (knnMethodContext.getMethodComponentContext().getParameters().size() != 0
                && knnMethodContext.getMethodComponentContext().getParameters().containsKey(METHOD_ENCODER_PARAMETER)) {
                Map<String, Object> methodComponentParams = knnMethodContext.getMethodComponentContext().getParameters();
                if ((methodComponentParams.get(METHOD_ENCODER_PARAMETER) instanceof MethodComponentContext)) {
                    MethodComponentContext encoderMethodComponentContext = (MethodComponentContext) methodComponentParams.get(
                        METHOD_ENCODER_PARAMETER
                    );
                    if (!ENCODER_FLAT.equals(encoderMethodComponentContext.getName())) {
                        throw new MapperParsingException("Encoder cannot be used with byte vector datatype");

                    }
                }
            }
        }

        try {
            Map<String, Object> methodParamsMap = knnEngine.getMethodAsMap(knnMethodContext);
            if (VectorDataType.BYTE.equals(vectorDataType) && methodParamsMap.containsKey(INDEX_DESCRIPTION_PARAMETER)) {
                String indexDescriptionValue = (String) methodParamsMap.get(INDEX_DESCRIPTION_PARAMETER);
                // String updatedIndexDescription = indexDescriptionValue.split(",")[0] + ",SQ8_direct_signed";
                String updatedIndexDescription = indexDescriptionValue.split(",")[0] + ",SQfp16";
                methodParamsMap.replace(INDEX_DESCRIPTION_PARAMETER, updatedIndexDescription);
            }
            ((Map<String, Object>) methodParamsMap.get("parameters")).remove("encoder");
            this.fieldType.putAttribute(PARAMETERS, XContentFactory.jsonBuilder().map(methodParamsMap).toString());
        } catch (IOException ioe) {
            throw new RuntimeException(String.format("Unable to create KNNVectorFieldMapper: %s", ioe));
        }

        this.fieldType.freeze();
    }
}
