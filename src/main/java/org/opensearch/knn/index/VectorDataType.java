/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.KnnVectorField;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.opensearch.index.mapper.ParametrizedFieldMapper;
import org.opensearch.knn.index.util.KNNEngine;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Locale;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.DEFAULT_VECTOR_DATA_TYPE_FIELD;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.LUCENE_NAME;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;

/**
 * Enum contains data_type of vectors and right now only supported for lucene engine in k-NN plugin.
 * We have two vector data_types, one is float (default) and the other one is byte.
 */
@AllArgsConstructor
public enum VectorDataType {
    BYTE("byte") {

        @Override
        public FieldType createKnnVectorFieldType(int dimension, VectorSimilarityFunction vectorSimilarityFunction) {
            return KnnByteVectorField.createFieldType(dimension, vectorSimilarityFunction);
        }
    },
    FLOAT("float") {

        @Override
        public FieldType createKnnVectorFieldType(int dimension, VectorSimilarityFunction vectorSimilarityFunction) {
            return KnnVectorField.createFieldType(dimension, vectorSimilarityFunction);
        }

    };

    @Getter
    private final String value;

    /**
     * Creates a KnnVectorFieldType based on the VectorDataType using the provided dimension and
     * VectorSimilarityFunction.
     *
     * @param dimension Dimension of the vector
     * @param vectorSimilarityFunction VectorSimilarityFunction for a given spaceType
     * @return FieldType
     */
    public abstract FieldType createKnnVectorFieldType(int dimension, VectorSimilarityFunction vectorSimilarityFunction);

    /**
     * @param knnEngine  KNNEngine
     * @return  DocValues FieldType of type Binary
     */
    public FieldType buildDocValuesFieldType(KNNEngine knnEngine) {
        FieldType field = new FieldType();
        field.putAttribute(KNN_ENGINE, knnEngine.getName());
        field.setDocValuesType(DocValuesType.BINARY);
        field.freeze();
        return field;
    }

    /**
     * @return  Set of names of all the supporting VectorDataTypes
     */
    protected static Set<String> getValues() {
        return Arrays.stream((VectorDataType.values())).map(VectorDataType::getValue).collect(Collectors.toCollection(HashSet::new));
    }

    /**
     * Validates if given VectorDataType is in the list of supported data types.
     * @param vectorDataType VectorDataType
     * @return  the same VectorDataType if it is in the supported values else throw exception.
     */
    public static VectorDataType get(String vectorDataType) {
        Objects.requireNonNull(
            vectorDataType,
            String.format(
                Locale.ROOT,
                "[{}] should not be null. Supported types are [{}]",
                VECTOR_DATA_TYPE_FIELD,
                String.join(",", getValues())
            )
        );
        return Arrays.stream(VectorDataType.values())
            .filter(e -> e.getValue().equalsIgnoreCase(vectorDataType))
            .findFirst()
            .orElseThrow(
                () -> new IllegalArgumentException(
                    String.format(
                        Locale.ROOT,
                        "Invalid value provided for [%s] field. Supported values are [%s]",
                        VECTOR_DATA_TYPE_FIELD,
                        String.join(",", getValues())
                    )
                )
            );
    }

    /**
     * Validate the float vector value and throw exception if it is not a number or not in the finite range.
     *
     * @param value  float vector value
     */
    public static void validateFloatVectorValue(float value) {
        if (Float.isNaN(value)) {
            throw new IllegalArgumentException("KNN vector values cannot be NaN");
        }

        if (Float.isInfinite(value)) {
            throw new IllegalArgumentException("KNN vector values cannot be infinity");
        }
    }

    /**
     * Validate the float vector value in the byte range if it is a finite number,
     * with no decimal values and in the byte range of [-128 to 127]. If not throw IllegalArgumentException.
     *
     * @param value  float value in byte range
     */
    public static void validateByteVectorValue(float value) {
        validateFloatVectorValue(value);
        if (value % 1 != 0) {
            throw new IllegalArgumentException(
                String.format(
                    Locale.ROOT,
                    "[%s] field was set as [%s] in index mapping. But, KNN vector values are floats instead of byte integers",
                    VECTOR_DATA_TYPE_FIELD,
                    VectorDataType.BYTE.getValue()
                )

            );
        }
        if ((int) value < Byte.MIN_VALUE || (int) value > Byte.MAX_VALUE) {
            throw new IllegalArgumentException(
                String.format(
                    Locale.ROOT,
                    "[%s] field was set as [%s] in index mapping. But, KNN vector values are not within in the byte range [%d, %d]",
                    VECTOR_DATA_TYPE_FIELD,
                    VectorDataType.BYTE.getValue(),
                    Byte.MIN_VALUE,
                    Byte.MAX_VALUE
                )
            );
        }
    }

    /**
     * Validate if the given vector size matches with the dimension provided in mapping.
     *
     * @param dimension dimension of vector
     * @param vectorSize size of the vector
     */
    public static void validateVectorDimension(int dimension, int vectorSize) {
        if (dimension != vectorSize) {
            String errorMessage = String.format(Locale.ROOT, "Vector dimension mismatch. Expected: %d, Given: %d", dimension, vectorSize);
            throw new IllegalArgumentException(errorMessage);
        }

    }

    /**
     * Validates and throws exception if data_type field is set in the index mapping
     * using any VectorDataType (other than float, which is default) with any engine (except lucene).
     *
     * @param knnMethodContext KNNMethodContext Parameter
     * @param vectorDataType VectorDataType Parameter
     */
    public static void validateVectorDataTypeWithEngine(
        ParametrizedFieldMapper.Parameter<KNNMethodContext> knnMethodContext,
        ParametrizedFieldMapper.Parameter<VectorDataType> vectorDataType
    ) {
        if (vectorDataType.getValue() == DEFAULT_VECTOR_DATA_TYPE_FIELD) {
            return;
        }
        if ((knnMethodContext.getValue() == null && KNNEngine.DEFAULT != KNNEngine.LUCENE)
            || knnMethodContext.getValue().getKnnEngine() != KNNEngine.LUCENE) {
            throw new IllegalArgumentException(
                String.format(
                    Locale.ROOT,
                    "[%s] field with value [%s] is only supported for [%s] engine",
                    VECTOR_DATA_TYPE_FIELD,
                    vectorDataType.getValue().getValue(),
                    LUCENE_NAME
                )
            );
        }
    }
}
