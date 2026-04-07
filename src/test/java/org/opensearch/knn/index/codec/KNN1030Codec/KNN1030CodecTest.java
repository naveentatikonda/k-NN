/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1030Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.codec.CustomCodec;
import org.opensearch.knn.index.codec.CustomCodecNoStoredFields;
import org.opensearch.knn.index.codec.KNNCodecTestCase;
import org.opensearch.knn.index.codec.KNNCodecVersion;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;

import java.util.Collections;
import java.util.Optional;
import java.util.function.Function;

import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.METHOD_FLAT;
import static org.opensearch.knn.index.engine.KNNEngine.LUCENE;

public class KNN1030CodecTest extends KNNCodecTestCase {

    @SneakyThrows
    public void testMultiFieldsKnnIndex() {
        testMultiFieldsKnnIndex(KNN1030Codec.builder().delegate(KNNCodecVersion.CURRENT_DEFAULT_DELEGATE).build());
    }

    @SneakyThrows
    public void testMultiFieldsKnnIndexCustomCodecWithStoredFields() {
        testMultiFieldsKnnIndex(KNN1030Codec.builder().delegate(new CustomCodec()).build());
    }

    @SneakyThrows
    public void testMultiFieldsKnnIndexCustomCodecWithoutStoredFields() {
        testMultiFieldsKnnIndex(KNN1030Codec.builder().delegate(new CustomCodecNoStoredFields()).build());
    }

    @SneakyThrows
    public void testBuildFromModelTemplate() {
        testBuildFromModelTemplate(KNN1030Codec.builder().delegate(KNNCodecVersion.CURRENT_DEFAULT_DELEGATE).build());
    }

    // Ensure that the codec is able to return the correct per field knn vectors format for codec
    public void testCodecSetsCustomPerFieldKnnVectorsFormat() {
        final Codec codec = new KNN1030Codec();
        assertTrue(codec.knnVectorsFormat() instanceof KNN1030PerFieldKnnVectorsFormat);
    }

    public void testFlatFormatResolver_returnsKNN1030ScalarQuantizedVectorsFormat() {
        KNNMethodContext flatMethodContext = new KNNMethodContext(
            LUCENE,
            SpaceType.L2,
            new MethodComponentContext(METHOD_FLAT, Collections.emptyMap())
        );

        MapperService mapperService = mock(MapperService.class);
        KNNVectorFieldType fieldType = new KNNVectorFieldType(
            "test_field",
            Collections.emptyMap(),
            org.opensearch.knn.index.VectorDataType.FLOAT,
            getMappingConfigForMethodMapping(flatMethodContext, 3)
        );
        when(mapperService.fieldType(eq("test_field"))).thenReturn(fieldType);

        KNN1030PerFieldKnnVectorsFormat format = new KNN1030PerFieldKnnVectorsFormat(Optional.of(mapperService));
        KnnVectorsFormat result = format.getKnnVectorsFormatForField("test_field");
        assertTrue(
            "Expected KNN1030ScalarQuantizedVectorsFormat but got " + result.getClass().getSimpleName(),
            result instanceof KNN1030ScalarQuantizedVectorsFormat
        );
    }

    // IMPORTANT: When this Codec is moved to a backwards Codec, this test needs to be removed, because it attempts to
    // write with a read-only codec, which will fail
    @SneakyThrows
    public void testKnnVectorIndex() {
        Function<MapperService, PerFieldKnnVectorsFormat> perFieldKnnVectorsFormatProvider = (
            mapperService) -> new KNN1030PerFieldKnnVectorsFormat(Optional.of(mapperService));

        Function<PerFieldKnnVectorsFormat, Codec> knnCodecProvider = (knnVectorFormat) -> KNN1030Codec.builder()
            .delegate(KNNCodecVersion.CURRENT_DEFAULT_DELEGATE)
            .knnVectorsFormat(knnVectorFormat)
            .build();

        testKnnVectorIndex(knnCodecProvider, perFieldKnnVectorsFormatProvider);
    }

}
