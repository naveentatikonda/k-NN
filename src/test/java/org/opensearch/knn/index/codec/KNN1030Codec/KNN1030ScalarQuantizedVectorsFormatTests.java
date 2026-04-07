/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1030Codec;

import org.apache.lucene.codecs.lucene103.Lucene103ScalarQuantizedVectorsFormat;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.engine.KNNEngine;

import static org.apache.lucene.codecs.lucene103.Lucene103ScalarQuantizedVectorsFormat.ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE;

public class KNN1030ScalarQuantizedVectorsFormatTests extends KNNTestCase {

    public void testToString_containsEncodingAndScorerInfo() {
        KNN1030ScalarQuantizedVectorsFormat format = new KNN1030ScalarQuantizedVectorsFormat(SINGLE_BIT_QUERY_NIBBLE);
        String str = format.toString();
        assertTrue("toString should contain class name", str.contains("KNN1030ScalarQuantizedVectorsFormat"));
        assertTrue("toString should contain encoding", str.contains(SINGLE_BIT_QUERY_NIBBLE.name()));
        assertTrue("toString should contain scorer", str.contains("ScalarQuantizedVectorScorer"));
    }

    public void testDefaultConstructor_usesSingleBitQueryNibble() {
        KNN1030ScalarQuantizedVectorsFormat format = new KNN1030ScalarQuantizedVectorsFormat();
        assertTrue(format.toString().contains(SINGLE_BIT_QUERY_NIBBLE.name()));
    }

    public void testGetMaxDimensions_returnsLuceneMax() {
        KNN1030ScalarQuantizedVectorsFormat format = new KNN1030ScalarQuantizedVectorsFormat();
        assertEquals(KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE), format.getMaxDimensions("any_field"));
    }

    public void testGetName_returnsClassName() {
        KNN1030ScalarQuantizedVectorsFormat format = new KNN1030ScalarQuantizedVectorsFormat();
        assertEquals("KNN1030ScalarQuantizedVectorsFormat", format.getName());
    }

    public void testConstructor_allEncodings() {
        for (Lucene103ScalarQuantizedVectorsFormat.ScalarEncoding encoding : Lucene103ScalarQuantizedVectorsFormat.ScalarEncoding
            .values()) {
            KNN1030ScalarQuantizedVectorsFormat format = new KNN1030ScalarQuantizedVectorsFormat(encoding);
            assertNotNull(format);
            assertTrue(format.toString().contains(encoding.name()));
        }
    }
}
