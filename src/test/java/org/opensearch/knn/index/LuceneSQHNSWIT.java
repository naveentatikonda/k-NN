/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import com.google.common.collect.ImmutableMap;
import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.junit.After;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.query.QueryBuilders;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.KNNQueryBuilder;

import java.io.IOException;
import java.util.List;
import java.util.Locale;

import static org.opensearch.knn.common.KNNConstants.COMPRESSION_LEVEL_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.LUCENE_SQ_BITS;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;

/**
 * End-to-end integration tests for Lucene {@code method=hnsw} with scalar quantization at
 * 2-bit (x16 compression) and 4-bit (x8 compression). Verifies mapping acceptance,
 * indexing, search correctness, updates/deletes, and filtered queries — mirrors the
 * shape of {@link LuceneSQFlatIT} but targets the HNSW method.
 */
public class LuceneSQHNSWIT extends KNNRestTestCase {

    private static final int DIMENSION = 128;
    private static final String PROPERTIES_FIELD = "properties";
    private static final String TYPE_FIELD = "type";
    private static final String KNN_VECTOR_TYPE = "knn_vector";
    private static final String DIMENSION_FIELD = "dimension";
    private static final String COLOR_FIELD_NAME = "color";
    private static final int M = 16;
    private static final int EF_CONSTRUCTION = 100;

    @After
    public final void cleanUp() throws IOException {
        deleteKNNIndex(INDEX_NAME);
    }

    // ---------- Happy-path: index + query ----------

    @SneakyThrows
    public void testIndexAndQuery_withHnswSQ_twoBit_l2() {
        createHnswSQIndex(SpaceType.L2, 2);
        indexTestDocs();

        // Query with all 1s — closest doc by L2 is doc 0 (all 1s), then doc 1 (all 2s), then doc 2 (all 3s).
        float[] queryVector = filledVector(DIMENSION, 1.0f);
        Response response = searchKNNIndex(INDEX_NAME, new KNNQueryBuilder(FIELD_NAME, queryVector, 3), 3);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);
        assertEquals(3, results.size());
        assertEquals("0", results.get(0).getDocId());
        assertEquals("1", results.get(1).getDocId());
        assertEquals("2", results.get(2).getDocId());
    }

    @SneakyThrows
    public void testIndexAndQuery_withHnswSQ_fourBit_l2() {
        createHnswSQIndex(SpaceType.L2, 4);
        indexTestDocs();

        float[] queryVector = filledVector(DIMENSION, 1.0f);
        Response response = searchKNNIndex(INDEX_NAME, new KNNQueryBuilder(FIELD_NAME, queryVector, 3), 3);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);
        assertEquals(3, results.size());
        assertEquals("0", results.get(0).getDocId());
        assertEquals("1", results.get(1).getDocId());
        assertEquals("2", results.get(2).getDocId());
    }

    @SneakyThrows
    public void testIndexAndQuery_withHnswSQ_twoBit_innerProduct() {
        createHnswSQIndex(SpaceType.INNER_PRODUCT, 2);
        indexTestDocs();

        // Query with all 1s — largest inner product is doc 4 (all 5s), doc 3 (all 4s), doc 2 (all 3s).
        float[] queryVector = filledVector(DIMENSION, 1.0f);
        Response response = searchKNNIndex(INDEX_NAME, new KNNQueryBuilder(FIELD_NAME, queryVector, 3), 3);
        String body = EntityUtils.toString(response.getEntity());
        List<KNNResult> results = parseSearchResponse(body, FIELD_NAME);
        assertEquals(3, results.size());
        assertEquals("4", results.get(0).getDocId());
        assertEquals("3", results.get(1).getDocId());
        assertEquals("2", results.get(2).getDocId());

        List<Float> scores = parseSearchResponseScore(body, FIELD_NAME);
        for (int i = 0; i < scores.size() - 1; i++) {
            assertTrue("Scores should be in descending order", scores.get(i) >= scores.get(i + 1));
        }
    }

    @SneakyThrows
    public void testIndexAndQuery_withHnswSQ_fourBit_innerProduct() {
        createHnswSQIndex(SpaceType.INNER_PRODUCT, 4);
        indexTestDocs();

        float[] queryVector = filledVector(DIMENSION, 1.0f);
        Response response = searchKNNIndex(INDEX_NAME, new KNNQueryBuilder(FIELD_NAME, queryVector, 3), 3);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);
        assertEquals(3, results.size());
        assertEquals("4", results.get(0).getDocId());
        assertEquals("3", results.get(1).getDocId());
        assertEquals("2", results.get(2).getDocId());
    }

    // ---------- Update / delete ----------

    @SneakyThrows
    public void testHnswSQ_withUpdateAndDelete_twoBit() {
        createHnswSQIndex(SpaceType.L2, 2);

        Float[] v1 = boxedVector(DIMENSION, 1.0f);
        Float[] v2 = boxedVector(DIMENSION, 2.0f);
        addKnnDoc(INDEX_NAME, "1", FIELD_NAME, v1);
        addKnnDoc(INDEX_NAME, "2", FIELD_NAME, v2);
        refreshIndex(INDEX_NAME);
        assertEquals(2, getDocCount(INDEX_NAME));

        updateKnnDoc(INDEX_NAME, "1", FIELD_NAME, v2);
        refreshIndex(INDEX_NAME);
        assertEquals(2, getDocCount(INDEX_NAME));

        deleteKnnDoc(INDEX_NAME, "2");
        refreshIndex(INDEX_NAME);
        assertEquals(1, getDocCount(INDEX_NAME));
    }

    // ---------- Filter ----------

    @SneakyThrows
    public void testHnswSQ_withFilter_fourBit() {
        // Build a mapping with an extra keyword field alongside the vector.
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD)
            .startObject(FIELD_NAME)
            .field(TYPE_FIELD, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD, DIMENSION)
            .startObject(KNNConstants.KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .field(KNNConstants.KNN_ENGINE, KNNEngine.LUCENE.getName())
            .startObject(PARAMETERS)
            .field(KNNConstants.METHOD_PARAMETER_M, M)
            .field(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION, EF_CONSTRUCTION)
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, ENCODER_SQ)
            .startObject(PARAMETERS)
            .field(LUCENE_SQ_BITS, 4)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .startObject(COLOR_FIELD_NAME)
            .field(TYPE_FIELD, "keyword")
            .endObject()
            .endObject()
            .endObject();
        createKnnIndex(INDEX_NAME, builder.toString());

        addKnnDocWithAttributes("0", filledVector(DIMENSION, 1.0f), ImmutableMap.of(COLOR_FIELD_NAME, "red"));
        addKnnDocWithAttributes("1", filledVector(DIMENSION, 2.0f), ImmutableMap.of(COLOR_FIELD_NAME, "green"));
        addKnnDocWithAttributes("2", filledVector(DIMENSION, 3.0f), ImmutableMap.of(COLOR_FIELD_NAME, "red"));
        addKnnDocWithAttributes("3", filledVector(DIMENSION, 4.0f), ImmutableMap.of(COLOR_FIELD_NAME, "green"));
        addKnnDocWithAttributes("4", filledVector(DIMENSION, 5.0f), ImmutableMap.of(COLOR_FIELD_NAME, "red"));
        refreshIndex(INDEX_NAME);

        float[] queryVector = filledVector(DIMENSION, 1.0f);
        Response response = searchKNNIndex(
            INDEX_NAME,
            new KNNQueryBuilder(FIELD_NAME, queryVector, 5, QueryBuilders.termQuery(COLOR_FIELD_NAME, "red")),
            5
        );
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);
        assertEquals("Should return the 3 red docs", 3, results.size());
        assertEquals("Nearest red to all-1s query is doc 0", "0", results.get(0).getDocId());
        for (KNNResult r : results) {
            assertTrue("Only red docs expected", r.getDocId().equals("0") || r.getDocId().equals("2") || r.getDocId().equals("4"));
        }
    }

    // ---------- Compression-level shortcut path ----------

    @SneakyThrows
    public void testHnswSQ_viaCompressionLevelParameter_x16() {
        // Users can request 2-bit SQ by setting compression_level: 16x without specifying the encoder.
        createHnswIndexViaCompressionLevel("16x");
        indexTestDocs();
        float[] queryVector = filledVector(DIMENSION, 1.0f);
        Response response = searchKNNIndex(INDEX_NAME, new KNNQueryBuilder(FIELD_NAME, queryVector, 3), 3);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);
        assertEquals(3, results.size());
        assertEquals("0", results.get(0).getDocId());
    }

    @SneakyThrows
    public void testHnswSQ_viaCompressionLevelParameter_x8() {
        createHnswIndexViaCompressionLevel("8x");
        indexTestDocs();
        float[] queryVector = filledVector(DIMENSION, 1.0f);
        Response response = searchKNNIndex(INDEX_NAME, new KNNQueryBuilder(FIELD_NAME, queryVector, 3), 3);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);
        assertEquals(3, results.size());
        assertEquals("0", results.get(0).getDocId());
    }

    // ---------- Validation / error cases ----------

    @SneakyThrows
    public void testHnswSQ_invalidBits_thenFail() {
        // Only 1, 2, 4, 7 are valid.
        for (int badBits : new int[] { 0, 3, 5, 6, 8 }) {
            expectThrows(
                ResponseException.class,
                String.format(Locale.ROOT, "Expected failure for bits=%d", badBits),
                () -> createKnnIndex(INDEX_NAME, buildSQMappingWithBits(badBits))
            );
        }
    }

    @SneakyThrows
    public void testHnswSQ_bitsAndCompressionMismatch_thenFail() {
        // bits=2 with compression_level=x8 should be rejected (bits=2 -> x16).
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD)
            .startObject(FIELD_NAME)
            .field(TYPE_FIELD, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD, DIMENSION)
            .field(COMPRESSION_LEVEL_PARAMETER, "8x")
            .startObject(KNNConstants.KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .field(KNNConstants.KNN_ENGINE, KNNEngine.LUCENE.getName())
            .startObject(PARAMETERS)
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, ENCODER_SQ)
            .startObject(PARAMETERS)
            .field(LUCENE_SQ_BITS, 2)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        expectThrows(ResponseException.class, () -> createKnnIndex(INDEX_NAME, builder.toString()));
    }

    // ---------- Helpers ----------

    private void createHnswSQIndex(SpaceType spaceType, int bits) throws IOException {
        createKnnIndex(INDEX_NAME, buildSQMappingWithBits(spaceType, bits));
    }

    private void createHnswIndexViaCompressionLevel(String compression) throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD)
            .startObject(FIELD_NAME)
            .field(TYPE_FIELD, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD, DIMENSION)
            .field(COMPRESSION_LEVEL_PARAMETER, compression)
            .startObject(KNNConstants.KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .field(KNNConstants.KNN_ENGINE, KNNEngine.LUCENE.getName())
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        createKnnIndex(INDEX_NAME, builder.toString());
    }

    private String buildSQMappingWithBits(int bits) throws IOException {
        return buildSQMappingWithBits(SpaceType.L2, bits);
    }

    private String buildSQMappingWithBits(SpaceType spaceType, int bits) throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD)
            .startObject(FIELD_NAME)
            .field(TYPE_FIELD, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD, DIMENSION)
            .startObject(KNNConstants.KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNNConstants.METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNNConstants.KNN_ENGINE, KNNEngine.LUCENE.getName())
            .startObject(PARAMETERS)
            .field(KNNConstants.METHOD_PARAMETER_M, M)
            .field(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION, EF_CONSTRUCTION)
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, ENCODER_SQ)
            .startObject(PARAMETERS)
            .field(LUCENE_SQ_BITS, bits)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        return builder.toString();
    }

    private void indexTestDocs() throws Exception {
        for (int i = 0; i < 5; i++) {
            addKnnDoc(INDEX_NAME, Integer.toString(i), FIELD_NAME, boxedVector(DIMENSION, i + 1));
        }
        refreshIndex(INDEX_NAME);
        assertEquals(5, getDocCount(INDEX_NAME));
    }

    private static float[] filledVector(int dimension, float value) {
        float[] v = new float[dimension];
        java.util.Arrays.fill(v, value);
        return v;
    }

    private static Float[] boxedVector(int dimension, float value) {
        Float[] v = new Float[dimension];
        java.util.Arrays.fill(v, value);
        return v;
    }
}
