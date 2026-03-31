/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.lucene104.Lucene104HnswScalarQuantizedVectorsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat.ScalarEncoding;
import org.apache.lucene.index.SegmentReadState;

import java.io.IOException;
import java.util.concurrent.ExecutorService;

/**
 * Custom HNSW scalar quantized vectors format that overrides the read path to use
 * {@link org.apache.lucene.store.DataAccessHint#NORMAL} for quantized vector data files ({@code .veb}).
 *
 * <p>Lucene's {@link Lucene104HnswScalarQuantizedVectorsFormat} delegates flat vector reading to
 * {@code Lucene104ScalarQuantizedVectorsReader}, which opens the {@code .veb} file with
 * {@link org.apache.lucene.store.DataAccessHint#RANDOM}. Since quantized vector data is read
 * sequentially during search, this format wraps the directory to force normal read advice,
 * improving I/O throughput.
 *
 * @see KNN1040ReadAdviceOverridingDirectory
 */
public class KNN1040HnswScalarQuantizedVectorsFormat extends Lucene104HnswScalarQuantizedVectorsFormat {

    public KNN1040HnswScalarQuantizedVectorsFormat(
        final ScalarEncoding scalarEncoding,
        final int maxConn,
        final int beamWidth,
        final int numMergeWorkers,
        final ExecutorService mergeExec
    ) {
        super(scalarEncoding, maxConn, beamWidth, numMergeWorkers, mergeExec);
    }

    @Override
    public KnnVectorsReader fieldsReader(final SegmentReadState state) throws IOException {
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
