/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.scorer;

import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorScorer;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.hnsw.RandomVectorScorer;

import java.io.IOException;

/**
 * A {@link Lucene104ScalarQuantizedVectorScorer} that wraps returned {@link RandomVectorScorer}
 * instances with {@link PrefetchableFlatVectorScorer.PrefetchableRandomVectorScorer} to enable
 * prefetch optimizations during search on the quantized scoring path.
 */
public class PrefetchableLucene104ScalarQuantizedVectorScorer extends Lucene104ScalarQuantizedVectorScorer {

    public PrefetchableLucene104ScalarQuantizedVectorScorer(final FlatVectorsScorer nonQuantizedDelegate) {
        super(nonQuantizedDelegate);
    }

    @Override
    public String toString() {
        return "PrefetchableLucene104ScalarQuantizedVectorScorer()";
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(
        VectorSimilarityFunction similarityFunction,
        KnnVectorValues vectorValues,
        float[] target
    ) throws IOException {
        return new PrefetchableFlatVectorScorer.PrefetchableRandomVectorScorer(
            (RandomVectorScorer.AbstractRandomVectorScorer) super.getRandomVectorScorer(similarityFunction, vectorValues, target)
        );
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(
        VectorSimilarityFunction similarityFunction,
        KnnVectorValues vectorValues,
        byte[] target
    ) throws IOException {
        return new PrefetchableFlatVectorScorer.PrefetchableRandomVectorScorer(
            (RandomVectorScorer.AbstractRandomVectorScorer) super.getRandomVectorScorer(similarityFunction, vectorValues, target)
        );
    }
}
