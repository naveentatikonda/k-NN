/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1030Codec;

import org.apache.lucene.codecs.lucene103.Lucene103ScalarQuantizedVectorsFormat.ScalarEncoding;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.engine.KNNEngine;

import java.util.concurrent.Executors;

import static org.apache.lucene.codecs.lucene103.Lucene103ScalarQuantizedVectorsFormat.ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE;

public class KNN1030HnswScalarQuantizedVectorsFormatTests extends KNNTestCase {

    public void testDefaultConstructor() {
        KNN1030HnswScalarQuantizedVectorsFormat format = new KNN1030HnswScalarQuantizedVectorsFormat();
        assertNotNull(format);
        String str = format.toString();
        assertTrue(str.contains("KNN1030HnswScalarQuantizedVectorsFormat"));
        assertTrue(str.contains(SINGLE_BIT_QUERY_NIBBLE.name()));
    }

    public void testCustomConstructor() {
        KNN1030HnswScalarQuantizedVectorsFormat format = new KNN1030HnswScalarQuantizedVectorsFormat(
            SINGLE_BIT_QUERY_NIBBLE,
            32,
            200,
            1,
            null
        );
        assertNotNull(format);
        assertEquals("KNN1030HnswScalarQuantizedVectorsFormat", format.getName());
        assertEquals(KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE), format.getMaxDimensions("any_field"));
    }

    public void testGetMaxDimensions_returnsLuceneMax() {
        KNN1030HnswScalarQuantizedVectorsFormat format = new KNN1030HnswScalarQuantizedVectorsFormat();
        assertEquals(KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE), format.getMaxDimensions("any_field"));
    }

    public void testGetName_returnsClassName() {
        KNN1030HnswScalarQuantizedVectorsFormat format = new KNN1030HnswScalarQuantizedVectorsFormat();
        assertEquals("KNN1030HnswScalarQuantizedVectorsFormat", format.getName());
    }

    public void testConstructor_invalidMaxConn_thenThrows() {
        expectThrows(
            IllegalArgumentException.class,
            () -> new KNN1030HnswScalarQuantizedVectorsFormat(SINGLE_BIT_QUERY_NIBBLE, 0, 100, 1, null)
        );
    }

    public void testConstructor_invalidBeamWidth_thenThrows() {
        expectThrows(
            IllegalArgumentException.class,
            () -> new KNN1030HnswScalarQuantizedVectorsFormat(SINGLE_BIT_QUERY_NIBBLE, 16, 0, 1, null)
        );
    }

    public void testConstructor_singleWorkerWithExecutor_thenThrows() {
        expectThrows(
            IllegalArgumentException.class,
            () -> new KNN1030HnswScalarQuantizedVectorsFormat(SINGLE_BIT_QUERY_NIBBLE, 16, 100, 1, Executors.newFixedThreadPool(1))
        );
    }

    public void testConstructor_allEncodings() {
        for (ScalarEncoding encoding : ScalarEncoding.values()) {
            KNN1030HnswScalarQuantizedVectorsFormat format = new KNN1030HnswScalarQuantizedVectorsFormat(encoding, 16, 100, 1, null);
            assertNotNull(format);
            assertTrue(format.toString().contains(encoding.name()));
        }
    }
}
