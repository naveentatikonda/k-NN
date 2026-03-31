/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.store.DataAccessHint;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;

import java.io.IOException;

/**
 * Directory wrapper that enforces {@link DataAccessHint#NORMAL} for {@code .veb} files when opening IndexInput.
 *
 * <p>Lucene's scalar quantized flat vector format ({@code Lucene104ScalarQuantizedVectorsReader}) opens
 * the quantized vector data file ({@code .veb}) with {@link DataAccessHint#RANDOM}, which disables
 * OS read-ahead prefetching. However, the quantized vector data is read sequentially during search
 * and merge operations. This wrapper intercepts {@link #openInput} and forces
 * {@link DataAccessHint#NORMAL} for {@code .veb} files to restore default OS read-ahead behavior,
 * improving sequential I/O throughput.
 *
 * <p>This is used by both the Lucene SQ flat path ({@link KNN1040ScalarQuantizedVectorsFormat})
 * and the Lucene SQ HNSW path ({@link KNN1040HnswScalarQuantizedVectorsFormat}).
 */
final class KNN1040ReadAdviceOverridingDirectory extends FilterDirectory {

    // Binary quantized vector data extension used by Lucene104ScalarQuantizedVectorsFormat
    private static final String QUANTIZED_VECTOR_DATA_EXTENSION = ".veb";

    KNN1040ReadAdviceOverridingDirectory(final Directory in) {
        super(in);
    }

    @Override
    public IndexInput openInput(final String fileName, final IOContext context) throws IOException {
        return in.openInput(fileName, overrideContextIfNeeded(fileName, context));
    }

    private static IOContext overrideContextIfNeeded(final String fileName, final IOContext defaultContext) {
        if (fileName.endsWith(QUANTIZED_VECTOR_DATA_EXTENSION)) {
            return defaultContext.withHints(DataAccessHint.NORMAL);
        }
        return defaultContext;
    }
}
