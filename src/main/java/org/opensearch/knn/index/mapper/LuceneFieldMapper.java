/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NonNull;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.KnnVectorField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.*;
import org.opensearch.common.Explicit;
import org.opensearch.index.mapper.ParseContext;
import org.opensearch.knn.index.KNNMethodContext;
import org.opensearch.knn.index.VectorField;
import org.opensearch.knn.index.util.KNNEngine;

import java.io.IOException;
import java.util.Map;
import java.util.Optional;

import static org.apache.lucene.index.VectorValues.MAX_DIMENSIONS;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;

/**
 * Field mapper for case when Lucene has been set as an engine.
 */
public class LuceneFieldMapper extends KNNVectorFieldMapper {

    private static final int LUCENE_MAX_DIMENSION = MAX_DIMENSIONS;

    /** FieldType used for initializing VectorField, which is used for creating binary doc values. **/
    private final FieldType vectorFieldType;
    // private final boolean isByteVector;
    private final String vectorDataType;

    LuceneFieldMapper(final CreateLuceneFieldMapperInput input) {
        super(
            input.getName(),
            input.getMappedFieldType(),
            input.getMultiFields(),
            input.getCopyTo(),
            input.getIgnoreMalformed(),
            input.isStored(),
            input.isHasDocValues()
        );

        // isByteVector = input.isByteVector();
        vectorDataType = input.getMappedFieldType().getVectorDataType();
        // isByteVector = true;
        this.knnMethod = input.getKnnMethodContext();
        final VectorSimilarityFunction vectorSimilarityFunction = this.knnMethod.getSpaceType().getVectorSimilarityFunction();

        final int dimension = input.getMappedFieldType().getDimension();
        if (dimension > LUCENE_MAX_DIMENSION) {
            throw new IllegalArgumentException(
                String.format(
                    "Dimension value cannot be greater than [%s] but got [%s] for vector [%s]",
                    LUCENE_MAX_DIMENSION,
                    dimension,
                    input.getName()
                )
            );
        }

        if (vectorDataType.equalsIgnoreCase("byte")) {
            System.out.println("Naveen: Inside isByteVector");
            this.fieldType = KnnByteVectorField.createFieldType(dimension, vectorSimilarityFunction);
        } else {
            System.out.println("Naveen: Not Inside isByteVector");
            this.fieldType = KnnVectorField.createFieldType(dimension, vectorSimilarityFunction);
        }

        // this.hasDocValues = true;
        if (this.hasDocValues) {
            this.vectorFieldType = buildDocValuesFieldType(
                this.knnMethod.getKnnEngine(),
                vectorDataType,
                input.mappedFieldType.getDimension()
            );
        } else {
            this.vectorFieldType = null;
        }
    }

    // private static FieldType buildDocValuesFieldType(KNNEngine knnEngine, String vectorDataType, int dimension) {
    // FieldType field = new FieldType();
    // if (vectorDataType.equals("byte")) {
    // field.setVectorAttributes(dimension, VectorEncoding.BYTE, field.vectorSimilarityFunction());
    // }
    //// field.setVectorAttributes(dimension, VectorEncoding.FLOAT32, field.vectorSimilarityFunction());
    // field.putAttribute(KNN_ENGINE, knnEngine.getName());
    // field.setDocValuesType(DocValuesType.BINARY);
    // field.freeze();
    // return field;
    // }

    private static FieldType buildDocValuesFieldType(KNNEngine knnEngine, String vectorDataType, int dimension) {
        FieldType field = null;
        if (vectorDataType.equals("byte")) {
//            IndexableFieldType indexableFieldType = new IndexableFieldType() {
//                @Override
//                public boolean stored() {
//                    return false;
//                }
//
//                @Override
//                public boolean tokenized() {
//                    return true;
//                }
//
//                @Override
//                public boolean storeTermVectors() {
//                    return false;
//                }
//
//                @Override
//                public boolean storeTermVectorOffsets() {
//                    return false;
//                }
//
//                @Override
//                public boolean storeTermVectorPositions() {
//                    return false;
//                }
//
//                @Override
//                public boolean storeTermVectorPayloads() {
//                    return false;
//                }
//
//                @Override
//                public boolean omitNorms() {
//                    return false;
//                }
//
//                @Override
//                public IndexOptions indexOptions() {
//                    return IndexOptions.NONE;
//                }
//
//                @Override
//                public DocValuesType docValuesType() {
//                    return DocValuesType.NONE;
//                }
//
//                @Override
//                public int pointDimensionCount() {
//                    return 0;
//                }
//
//                @Override
//                public int pointIndexDimensionCount() {
//                    return 0;
//                }
//
//                @Override
//                public int pointNumBytes() {
//                    return 0;
//                }
//
//                @Override
//                public int vectorDimension() {
//                    return 0;
//                }
//
//                @Override
//                public VectorEncoding vectorEncoding() {
//                    return VectorEncoding.BYTE;
//                }
//
//                @Override
//                public VectorSimilarityFunction vectorSimilarityFunction() {
//                    return VectorSimilarityFunction.EUCLIDEAN;
//                }
//
//                @Override
//                public Map<String, String> getAttributes() {
//                    return null;
//                }
//            };
//            field = new FieldType(indexableFieldType);
            field = new FieldType();
            // field.setVectorAttributes(dimension, VectorEncoding.BYTE, field.vectorSimilarityFunction());
        }
        // field.setVectorAttributes(dimension, VectorEncoding.FLOAT32, field.vectorSimilarityFunction());
        else {
            field = new FieldType();
        }
        field.putAttribute(KNN_ENGINE, knnEngine.getName());
        field.setDocValuesType(DocValuesType.BINARY);
        field.freeze();
        return field;
    }

    @Override
    protected void parseCreateField(ParseContext context, int dimension) throws IOException {

        validateIfKNNPluginEnabled();
        validateIfCircuitBreakerIsNotTriggered();

        if (vectorDataType.equalsIgnoreCase("byte")) {
            Optional<byte[]> arrayOptional = getBytesFromContext(context, dimension);
            if (arrayOptional.isEmpty()) {
                return;
            }
            final byte[] array = arrayOptional.get();
            KnnByteVectorField point = new KnnByteVectorField(name(), array, fieldType);
            context.doc().add(point);
            if (fieldType.stored()) {
                context.doc().add(new StoredField(name(), point.toString()));
            }

            if (hasDocValues && vectorFieldType != null) {
                context.doc().add(new VectorField(name(), array, vectorFieldType));
            }
        } else {
            Optional<float[]> arrayOptional = getFloatsFromContext(context, dimension);
            if (arrayOptional.isEmpty()) {
                return;
            }
            final float[] array = arrayOptional.get();
            KnnVectorField point = new KnnVectorField(name(), array, fieldType);
            context.doc().add(point);
            if (fieldType.stored()) {
                context.doc().add(new StoredField(name(), point.toString()));
            }

            if (hasDocValues && vectorFieldType != null) {
                context.doc().add(new VectorField(name(), array, vectorFieldType));
            }
        }
        context.path().remove();
    }

    @Override
    void updateEngineStats() {
        KNNEngine.LUCENE.setInitialized(true);
    }

    @AllArgsConstructor
    @lombok.Builder
    @Getter
    static class CreateLuceneFieldMapperInput {
        @NonNull
        String name;
        @NonNull
        KNNVectorFieldType mappedFieldType;
        @NonNull
        MultiFields multiFields;
        @NonNull
        CopyTo copyTo;
        @NonNull
        Explicit<Boolean> ignoreMalformed;
        boolean stored;
        boolean hasDocValues;
        String vectorDataType;
        @NonNull
        KNNMethodContext knnMethodContext;
    }
}
