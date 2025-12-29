/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.lucene;

import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.Weight;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.FloatPoint;
import org.mockito.Mock;
import org.mockito.MockedStatic;
import org.opensearch.knn.index.query.common.QueryUtils;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.MockitoAnnotations.openMocks;

public class LuceneEngineKnnVectorQueryTests extends OpenSearchTestCase {

    @Mock
    IndexSearcher indexSearcher;

    @Mock
    Query luceneQuery;

    @Mock
    Weight weight;

    @Mock
    QueryVisitor queryVisitor;

    LuceneEngineKnnVectorQuery objectUnderTest;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        openMocks(this);
        objectUnderTest = new LuceneEngineKnnVectorQuery(luceneQuery, 10, 5);
        when(luceneQuery.rewrite(any(IndexSearcher.class))).thenReturn(luceneQuery);
        when(luceneQuery.createWeight(any(IndexSearcher.class), any(ScoreMode.class), anyFloat())).thenReturn(weight);
        when(luceneQuery.toString()).thenReturn("mocked query string");
    }

    public void testRewrite() {
        Query result1 = objectUnderTest.rewrite(indexSearcher);
        Query result2 = objectUnderTest.rewrite(indexSearcher);
        Query result3 = objectUnderTest.rewrite(indexSearcher);
        assertEquals(objectUnderTest, result1);
        assertEquals(objectUnderTest, result2);
        assertEquals(objectUnderTest, result3);
        verifyNoInteractions(luceneQuery);
    }

    public void testVisit() {
        objectUnderTest.visit(queryVisitor);
        verify(queryVisitor).visitLeaf(objectUnderTest);
    }

    public void testEquals() {
        LuceneEngineKnnVectorQuery mainQuery = new LuceneEngineKnnVectorQuery(luceneQuery, 10, 5);
        LuceneEngineKnnVectorQuery otherQuery = new LuceneEngineKnnVectorQuery(luceneQuery, 10, 5);
        assertEquals(mainQuery, otherQuery);
        assertEquals(mainQuery, mainQuery);
        assertNotEquals(mainQuery, null);
        assertNotEquals(mainQuery, new Object());
        LuceneEngineKnnVectorQuery otherQuery2 = new LuceneEngineKnnVectorQuery(null, 10, 5);
        assertNotEquals(mainQuery, otherQuery2);
        LuceneEngineKnnVectorQuery otherQuery3 = new LuceneEngineKnnVectorQuery(luceneQuery, 15, 5);
        assertNotEquals(mainQuery, otherQuery3);
        LuceneEngineKnnVectorQuery otherQuery4 = new LuceneEngineKnnVectorQuery(luceneQuery, 10, 8);
        assertNotEquals(mainQuery, otherQuery4);
    }

    public void testHashCode() {
        LuceneEngineKnnVectorQuery mainQuery1 = new LuceneEngineKnnVectorQuery(luceneQuery, 10, 5);
        LuceneEngineKnnVectorQuery mainQuery2 = new LuceneEngineKnnVectorQuery(luceneQuery, 10, 5);
        LuceneEngineKnnVectorQuery differentQuery = new LuceneEngineKnnVectorQuery(luceneQuery, 15, 5);

        assertEquals(mainQuery1.hashCode(), mainQuery2.hashCode());
        assertNotEquals(mainQuery1.hashCode(), differentQuery.hashCode());
    }

    public void testToString() {
        LuceneEngineKnnVectorQuery mainQuery = new LuceneEngineKnnVectorQuery(luceneQuery, 10, 5);
        String expected = "LuceneEngineKnnVectorQuery[luceneK=10, k=5, query=mocked query string]";
        assertEquals(expected, mainQuery.toString());
    }

    public void testCreateWeightWithoutReducing() throws IOException {
        // Test luceneK == k, should return original query
        LuceneEngineKnnVectorQuery queryNoReduce = new LuceneEngineKnnVectorQuery(luceneQuery, 5, 5);
        Weight actualWeight = queryNoReduce.createWeight(indexSearcher, ScoreMode.TOP_DOCS, 1.0f);
        verify(luceneQuery, times(1)).rewrite(indexSearcher);
        verify(luceneQuery, times(1)).createWeight(indexSearcher, ScoreMode.TOP_DOCS, 1.0f);
        assertEquals(weight, actualWeight);
    }

    public void testCreateWeightWithReduceToTopK() throws Exception {
        // Create a real directory with multiple segments to get multiple LeafReaderContexts
        Directory directory = new ByteBuffersDirectory();
        IndexWriterConfig config = new IndexWriterConfig();
        try (IndexWriter writer = new IndexWriter(directory, config)) {
            // Add documents to first segment
            Document doc1 = new Document();
            doc1.add(new FloatPoint("vector", 1.0f, 2.0f));
            writer.addDocument(doc1);
            writer.flush(); // Force segment creation

            // Add documents to second segment
            Document doc2 = new Document();
            doc2.add(new FloatPoint("vector", 3.0f, 4.0f));
            writer.addDocument(doc2);
            writer.commit();
        }

        DirectoryReader reader = DirectoryReader.open(directory);
        List<LeafReaderContext> leaves = reader.leaves();
        assertEquals(2, leaves.size()); // Ensure we have 2 segments

        when(indexSearcher.getIndexReader()).thenReturn(reader);

        try (MockedStatic<QueryUtils> queryUtilsMock = mockStatic(QueryUtils.class)) {

            QueryUtils mockQueryUtils = mock(QueryUtils.class);
            Query mockDocAndScoreQuery = mock(Query.class);
            Weight mockDocAndScoreWeight = mock(Weight.class);

            // Mock doSearch to return leaf-relative doc IDs from both segments
            Map<Integer, Float> leafResult1 = new HashMap<>();
            leafResult1.put(0, 1.0f); // leaf-relative doc ID 0 with score 1.0
            Map<Integer, Float> leafResult2 = new HashMap<>();
            leafResult2.put(0, 0.8f); // leaf-relative doc ID 0 with score 0.8
            List<Map<Integer, Float>> leafResults = Arrays.asList(leafResult1, leafResult2);

            queryUtilsMock.when(QueryUtils::getInstance).thenReturn(mockQueryUtils);
            when(mockQueryUtils.doSearch(eq(indexSearcher), eq(leaves), eq(weight))).thenReturn(leafResults);
            when(mockQueryUtils.createDocAndScoreQuery(any(), any())).thenReturn(mockDocAndScoreQuery);
            when(mockDocAndScoreQuery.createWeight(any(), any(), anyFloat())).thenReturn(mockDocAndScoreWeight);

            LuceneEngineKnnVectorQuery queryWithReduce = new LuceneEngineKnnVectorQuery(luceneQuery, 10, 3);
            Weight reducedWeight = queryWithReduce.createWeight(indexSearcher, ScoreMode.TOP_DOCS, 1.0f);

            verify(luceneQuery).createWeight(indexSearcher, ScoreMode.TOP_DOCS, 1.0f);
            verify(mockQueryUtils).doSearch(eq(indexSearcher), eq(leaves), eq(weight));
            verify(mockQueryUtils).createDocAndScoreQuery(eq(reader), any(TopDocs.class));
            assertEquals(mockDocAndScoreWeight, reducedWeight);
        }

        reader.close();
        directory.close();
    }
}
