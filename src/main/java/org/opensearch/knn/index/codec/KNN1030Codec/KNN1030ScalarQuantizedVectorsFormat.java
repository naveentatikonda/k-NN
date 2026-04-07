/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1030Codec;

import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.lucene103.Lucene103ScalarQuantizedVectorsFormat;
import org.apache.lucene.codecs.lucene103.Lucene103ScalarQuantizedVectorsReader;
import org.apache.lucene.codecs.lucene103.Lucene103ScalarQuantizedVectorsWriter;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.memoryoptsearch.faiss.FlatVectorsScorerProvider;

import java.io.IOException;

/**
 * A {@link Lucene103ScalarQuantizedVectorsFormat} that uses a {@link KNN1030ScalarQuantizedVectorScorer}
 * to take advantage of SIMD-accelerated scoring during search.
 */
public class KNN1030ScalarQuantizedVectorsFormat extends Lucene103ScalarQuantizedVectorsFormat {

    private static final KNN1030ScalarQuantizedVectorScorer KNN_1040_SCALAR_QUANTIZED_VECTOR_SCORER = FlatVectorsScorerProvider
        .getKNN1030ScalarQuantizedVectorScorer(FlatVectorsScorerProvider.getLucene99FlatVectorsScorer());

    // Must use the default Lucene scorer here, not KNN_1040_SCALAR_QUANTIZED_VECTOR_SCORER.
    // KNN1030ScalarQuantizedVectorScorer.getRandomVectorScorer(float[]) always assumes quantized
    // vectors and will fail (NPE/exception) when called with raw OffHeapFloatVectorValues.
    private static final Lucene99FlatVectorsFormat RAW_VECTOR_FORMAT = new Lucene99FlatVectorsFormat(
        FlatVectorsScorerProvider.getLucene99FlatVectorsScorer()
    );

    private final ScalarEncoding encoding;

    public KNN1030ScalarQuantizedVectorsFormat() {
        this(ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE);
    }

    public KNN1030ScalarQuantizedVectorsFormat(final ScalarEncoding encoding) {
        super(encoding);
        this.encoding = encoding;
    }

    @Override
    public FlatVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
        return new Lucene103ScalarQuantizedVectorsWriter(
            state,
            encoding,
            RAW_VECTOR_FORMAT.fieldsWriter(state),
            KNN_1040_SCALAR_QUANTIZED_VECTOR_SCORER
        );
    }

    @Override
    public String toString() {
        return String.format(
            "%s(encoding=%s, scorer=%s, rawVectorFormat=%s)",
            getClass().getSimpleName(),
            encoding,
            KNN_1040_SCALAR_QUANTIZED_VECTOR_SCORER,
            RAW_VECTOR_FORMAT
        );
    }

    @Override
    public FlatVectorsReader fieldsReader(SegmentReadState state) throws IOException {
        return new Lucene103ScalarQuantizedVectorsReader(
            state,
            RAW_VECTOR_FORMAT.fieldsReader(state),
            KNN_1040_SCALAR_QUANTIZED_VECTOR_SCORER
        );
    }

    @Override
    public int getMaxDimensions(String fieldName) {
        return KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE);
    }

    @Override
    public String getName() {
        return getClass().getSimpleName();
    }
}
