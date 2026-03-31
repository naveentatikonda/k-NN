/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat;
import org.apache.lucene.index.SegmentReadState;

import java.io.IOException;

/**
 * Custom scalar quantized flat vectors format that overrides the read path to use
 * {@link org.apache.lucene.store.DataAccessHint#NORMAL} for quantized vector data files ({@code .veb}).
 *
 * <p>Lucene's {@link Lucene104ScalarQuantizedVectorsFormat} opens the {@code .veb} file with
 * {@link org.apache.lucene.store.DataAccessHint#RANDOM}, which disables OS read-ahead. Since
 * quantized vector data is read sequentially during search, this format wraps the directory
 * to force normal read advice, improving I/O throughput.
 *
 * @see KNN1040ReadAdviceOverridingDirectory
 */
public class KNN1040ScalarQuantizedVectorsFormat extends Lucene104ScalarQuantizedVectorsFormat {

    public KNN1040ScalarQuantizedVectorsFormat(final ScalarEncoding scalarEncoding) {
        super(scalarEncoding);
    }

    @Override
    public FlatVectorsReader fieldsReader(final SegmentReadState state) throws IOException {
        final SegmentReadState stateWithOverriddenDirectory = new SegmentReadState(
            new KNN1040ReadAdviceOverridingDirectory(state.directory),
            state.segmentInfo,
            state.fieldInfos,
            state.context,
            state.segmentSuffix
        );
        return super.fieldsReader(stateWithOverriddenDirectory);
    }
}
