/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1030Codec;

import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.lucene103.Lucene103HnswScalarQuantizedVectorsFormat;
import org.apache.lucene.codecs.lucene103.Lucene103ScalarQuantizedVectorsFormat.ScalarEncoding;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsWriter;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.search.TaskExecutor;
import org.opensearch.knn.index.engine.KNNEngine;

import java.io.IOException;
import java.util.concurrent.ExecutorService;

import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat.DEFAULT_BEAM_WIDTH;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat.DEFAULT_MAX_CONN;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat.DEFAULT_NUM_MERGE_WORKER;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat.HNSW_GRAPH_THRESHOLD;

/**
 * HNSW + scalar quantization format that extends {@link Lucene103HnswScalarQuantizedVectorsFormat}
 * and overrides flat vector operations to use {@link KNN1030ScalarQuantizedVectorsFormat},
 * inheriting its SIMD-accelerated {@link KNN1030ScalarQuantizedVectorScorer} for graph traversal scoring.
 */
public class KNN1030HnswScalarQuantizedVectorsFormat extends Lucene103HnswScalarQuantizedVectorsFormat {

    private final int maxConn;
    private final int beamWidth;
    private final int tinySegmentsThreshold;
    private final int numMergeWorkers;
    private final TaskExecutor mergeExec;
    private final KNN1030ScalarQuantizedVectorsFormat flatVectorsFormat;

    public KNN1030HnswScalarQuantizedVectorsFormat() {
        this(ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE, DEFAULT_MAX_CONN, DEFAULT_BEAM_WIDTH, DEFAULT_NUM_MERGE_WORKER, null);
    }

    public KNN1030HnswScalarQuantizedVectorsFormat(
        ScalarEncoding encoding,
        int maxConn,
        int beamWidth,
        int numMergeWorkers,
        ExecutorService mergeExec
    ) {
        this(encoding, maxConn, beamWidth, numMergeWorkers, mergeExec, HNSW_GRAPH_THRESHOLD);
    }

    public KNN1030HnswScalarQuantizedVectorsFormat(
        ScalarEncoding encoding,
        int maxConn,
        int beamWidth,
        int numMergeWorkers,
        ExecutorService mergeExec,
        int tinySegmentsThreshold
    ) {
        super(encoding, maxConn, beamWidth, numMergeWorkers, mergeExec, tinySegmentsThreshold);
        this.maxConn = maxConn;
        this.beamWidth = beamWidth;
        this.tinySegmentsThreshold = tinySegmentsThreshold;
        this.numMergeWorkers = numMergeWorkers;
        this.mergeExec = mergeExec != null ? new TaskExecutor(mergeExec) : null;
        this.flatVectorsFormat = new KNN1030ScalarQuantizedVectorsFormat(encoding);
    }

    @Override
    public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
        return new Lucene99HnswVectorsWriter(
            state,
            maxConn,
            beamWidth,
            flatVectorsFormat.fieldsWriter(state),
            numMergeWorkers,
            mergeExec,
            tinySegmentsThreshold
        );
    }

    @Override
    public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
        return new Lucene99HnswVectorsReader(state, flatVectorsFormat.fieldsReader(state));
    }

    @Override
    public int getMaxDimensions(String fieldName) {
        return KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE);
    }

    @Override
    public String getName() {
        return getClass().getSimpleName();
    }

    @Override
    public String toString() {
        return String.format(
            "%s(maxConn=%d, beamWidth=%d, tinySegmentsThreshold=%d, flatVectorFormat=%s)",
            getClass().getSimpleName(),
            maxConn,
            beamWidth,
            tinySegmentsThreshold,
            flatVectorsFormat
        );
    }
}
