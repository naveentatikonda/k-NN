/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.scorer;

import junit.framework.TestCase;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat;
import org.apache.lucene.codecs.lucene104.QuantizedByteVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;

import java.io.IOException;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Tests for {@link PrefetchableLucene104ScalarQuantizedVectorScorer}.
 */
public class PrefetchableLucene104ScalarQuantizedVectorScorerTests extends TestCase {

    private final FlatVectorsScorer delegate = mock(FlatVectorsScorer.class);
    private final PrefetchableLucene104ScalarQuantizedVectorScorer scorer = new PrefetchableLucene104ScalarQuantizedVectorScorer(delegate);

    public void testToString_containsClassName() {
        assertEquals("PrefetchableLucene104ScalarQuantizedVectorScorer()", scorer.toString());
    }

    /**
     * When vectorValues is a QuantizedByteVectorValues, the super takes the quantized path and
     * returns an AbstractRandomVectorScorer. Verifies it is wrapped in a PrefetchableRandomVectorScorer.
     */
    public void testGetRandomVectorScorer_floatTarget_quantizedValues_wrapsSuperResultInPrefetchableScorer() throws IOException {
        int dims = 8;
        float[] target = new float[dims];
        float[] centroid = new float[dims];
        for (int i = 0; i < dims; i++) {
            target[i] = 0.1f * i;
            centroid[i] = 0.0f;
        }

        QuantizedByteVectorValues quantizedValues = mock(QuantizedByteVectorValues.class);
        when(quantizedValues.dimension()).thenReturn(dims);
        when(quantizedValues.size()).thenReturn(1);
        when(quantizedValues.getScalarEncoding()).thenReturn(Lucene104ScalarQuantizedVectorsFormat.ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE);
        when(quantizedValues.getQuantizer()).thenReturn(new OptimizedScalarQuantizer(VectorSimilarityFunction.EUCLIDEAN));
        when(quantizedValues.getCentroid()).thenReturn(centroid);

        RandomVectorScorer result = scorer.getRandomVectorScorer(VectorSimilarityFunction.EUCLIDEAN, quantizedValues, target);

        assertNotNull(result);
        assertTrue(
            "Expected PrefetchableRandomVectorScorer but got " + result.getClass().getSimpleName(),
            result instanceof PrefetchableFlatVectorScorer.PrefetchableRandomVectorScorer
        );
    }

    /**
     * When vectorValues is not QuantizedByteVectorValues, the super falls back to nonQuantizedDelegate.
     * Verifies the delegate's result is wrapped in a PrefetchableRandomVectorScorer.
     */
    public void testGetRandomVectorScorer_floatTarget_nonQuantizedValues_wrapsDelegateResultInPrefetchableScorer() throws IOException {
        KnnVectorValues vectorValues = mock(KnnVectorValues.class);
        float[] target = new float[] { 1.0f, 2.0f };

        RandomVectorScorer.AbstractRandomVectorScorer delegateScorer = new RandomVectorScorer.AbstractRandomVectorScorer(vectorValues) {
            @Override
            public float score(int node) {
                return 0;
            }
        };
        when(delegate.getRandomVectorScorer(VectorSimilarityFunction.EUCLIDEAN, vectorValues, target)).thenReturn(delegateScorer);

        RandomVectorScorer result = scorer.getRandomVectorScorer(VectorSimilarityFunction.EUCLIDEAN, vectorValues, target);

        assertNotNull(result);
        assertTrue(
            "Expected PrefetchableRandomVectorScorer but got " + result.getClass().getSimpleName(),
            result instanceof PrefetchableFlatVectorScorer.PrefetchableRandomVectorScorer
        );
        verify(delegate).getRandomVectorScorer(VectorSimilarityFunction.EUCLIDEAN, vectorValues, target);
    }

    /**
     * When vectorValues is not QuantizedByteVectorValues, the super falls back to the delegate.
     * Verifies the delegate's result is wrapped in a PrefetchableRandomVectorScorer for byte targets.
     */
    public void testGetRandomVectorScorer_byteTarget_wrapsDelegateResultInPrefetchableScorer() throws IOException {
        KnnVectorValues vectorValues = mock(KnnVectorValues.class);
        byte[] target = new byte[] { 1, 2 };
        when(vectorValues.dimension()).thenReturn(target.length);

        RandomVectorScorer.AbstractRandomVectorScorer delegateScorer = new RandomVectorScorer.AbstractRandomVectorScorer(vectorValues) {
            @Override
            public float score(int node) {
                return 0;
            }
        };
        when(delegate.getRandomVectorScorer(VectorSimilarityFunction.EUCLIDEAN, vectorValues, target)).thenReturn(delegateScorer);

        RandomVectorScorer result = scorer.getRandomVectorScorer(VectorSimilarityFunction.EUCLIDEAN, vectorValues, target);

        assertNotNull(result);
        assertTrue(
            "Expected PrefetchableRandomVectorScorer but got " + result.getClass().getSimpleName(),
            result instanceof PrefetchableFlatVectorScorer.PrefetchableRandomVectorScorer
        );
        verify(delegate).getRandomVectorScorer(VectorSimilarityFunction.EUCLIDEAN, vectorValues, target);
    }

    /**
     * Verifies that getRandomVectorScorerSupplier delegates unchanged to the underlying scorer.
     */
    public void testGetRandomVectorScorerSupplier_delegatesToSuper() throws IOException {
        KnnVectorValues vectorValues = mock(KnnVectorValues.class);
        RandomVectorScorerSupplier expectedSupplier = mock(RandomVectorScorerSupplier.class);

        when(delegate.getRandomVectorScorerSupplier(VectorSimilarityFunction.EUCLIDEAN, vectorValues)).thenReturn(expectedSupplier);

        RandomVectorScorerSupplier result = scorer.getRandomVectorScorerSupplier(VectorSimilarityFunction.EUCLIDEAN, vectorValues);

        assertSame(expectedSupplier, result);
        verify(delegate).getRandomVectorScorerSupplier(VectorSimilarityFunction.EUCLIDEAN, vectorValues);
    }
}
