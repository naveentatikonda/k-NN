/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.DataAccessHint;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;
import org.opensearch.knn.index.engine.KNNEngine;

import java.io.IOException;

/**
 * This is a Vector format that will be used for Native engines like Faiss and Nmslib for reading and writing vector
 * related data structures.
 */
@Log4j2
public class NativeEngines990KnnVectorsFormat extends KnnVectorsFormat {
    /** The format for storing, reading, merging vectors on disk */
    private static FlatVectorsFormat flatVectorsFormat;
    private static final String FORMAT_NAME = "NativeEngines990KnnVectorsFormat";
    private static int approximateThreshold;
    private final NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory;

    public NativeEngines990KnnVectorsFormat() {
        this(new Lucene99FlatVectorsFormat(new DefaultFlatVectorScorer()));
    }

    public NativeEngines990KnnVectorsFormat(int approximateThreshold) {
        this(new Lucene99FlatVectorsFormat(new DefaultFlatVectorScorer()), approximateThreshold);
    }

    public NativeEngines990KnnVectorsFormat(final FlatVectorsFormat flatVectorsFormat) {
        this(flatVectorsFormat, KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_DEFAULT_VALUE);
    }

    public NativeEngines990KnnVectorsFormat(final FlatVectorsFormat flatVectorsFormat, int approximateThreshold) {
        this(flatVectorsFormat, approximateThreshold, new NativeIndexBuildStrategyFactory());
    }

    public NativeEngines990KnnVectorsFormat(
        final FlatVectorsFormat flatVectorsFormat,
        int approximateThreshold,
        final NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory
    ) {
        super(FORMAT_NAME);
        NativeEngines990KnnVectorsFormat.flatVectorsFormat = flatVectorsFormat;
        NativeEngines990KnnVectorsFormat.approximateThreshold = approximateThreshold;
        this.nativeIndexBuildStrategyFactory = nativeIndexBuildStrategyFactory;
    }

    /**
     * Returns a {@link KnnVectorsWriter} to write the vectors to the index.
     *
     * @param state {@link SegmentWriteState}
     */
    @Override
    public KnnVectorsWriter fieldsWriter(final SegmentWriteState state) throws IOException {
        return new NativeEngines990KnnVectorsWriter(
            state,
            flatVectorsFormat.fieldsWriter(state),
            approximateThreshold,
            nativeIndexBuildStrategyFactory
        );
    }

    /**
     * Returns a {@link KnnVectorsReader} to read the vectors from the index.
     *
     * @param state {@link SegmentReadState}
     */
    @Override
    public KnnVectorsReader fieldsReader(final SegmentReadState state) throws IOException {
        final SegmentReadState stateWithCustomDirectory = new SegmentReadState(
            new FlagsOverridingMMapDirectory(state.directory),
            state.segmentInfo,
            state.fieldInfos,
            state.context,
            state.segmentSuffix
        );
        return new NativeEngines990KnnVectorsReader(state, flatVectorsFormat.fieldsReader(stateWithCustomDirectory));
    }

    /**
     * @param s
     * @return
     */
    @Override
    public int getMaxDimensions(String s) {
        return KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE);
    }

    @Override
    public String toString() {
        return "NativeEngines99KnnVectorsFormat(name="
            + this.getClass().getSimpleName()
            + ", flatVectorsFormat="
            + flatVectorsFormat
            + ", approximateThreshold="
            + approximateThreshold
            + ")";
    }

    // Directory wrapper to enforce `ReadAdvice.NORMAL` for `.vec` files when opening IndexInput.
    // Lucene’s flat vector format opens IndexInput with `ReadAdvice.RANDOM`, which introduces
    // read overhead for exact search. Since the implementation is final and cannot
    // be easily overridden for us, this wrapper intercepts `openInput` and forces `ReadAdvice.NORMAL`
    // to restore default OS read-ahead behavior which can boost exact search.
    private static final class FlagsOverridingMMapDirectory extends FilterDirectory {
        // See Lucene99FlatVectorsFormat.VECTOR_DATA_EXTENSION
        private static final String VECTOR_DATA_EXTENSION = ".vec";

        private FlagsOverridingMMapDirectory(final Directory in) {
            super(in);
        }

        @Override
        public IndexInput openInput(final String fileName, final IOContext context) throws IOException {
            return in.openInput(fileName, createNormalIOContext(fileName, context));
        }

        private IOContext createNormalIOContext(final String fileName, final IOContext defaultContext) {
            if (fileName.endsWith(VECTOR_DATA_EXTENSION)) {
                // For all .vec file, we force SEQUENTIAL access to disable random access flags.
                return IOContext.DEFAULT.withHints(DataAccessHint.SEQUENTIAL);
            }

            // Return what its was given.
            return defaultContext;
        }
    }
}
