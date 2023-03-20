/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.action;

import org.junit.Test;
import org.opensearch.client.Request;
import org.opensearch.client.ResponseException;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.plugin.KNNPlugin;
import org.opensearch.rest.RestRequest;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;

import static org.opensearch.knn.common.KNNConstants.CLEAR_CACHE;

/**
 * Integration tests to validate ClearCache API
 */

public class RestClearCacheHandlerIT extends KNNRestTestCase {
    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSIONS = 2;

    // @Test(expected = ResponseException.class)
    public void testNonExistentIndex() throws IOException {
        String nonExistentIndex = "non-existent-index";

        String restURI = String.join("/", KNNPlugin.KNN_BASE_URI, CLEAR_CACHE, nonExistentIndex);
        Request request = new Request(RestRequest.Method.POST.name(), restURI);

        ResponseException ex = expectThrows(ResponseException.class, () -> client().performRequest(request));
        assertTrue(ex.getMessage().contains(nonExistentIndex));
    }

    @Test(expected = ResponseException.class)
    public void testNotKnnIndex() throws IOException {
        String notKNNIndex = "not-KNN-index";
        createIndex(notKNNIndex, Settings.EMPTY);

        String restURI = String.join("/", KNNPlugin.KNN_BASE_URI, CLEAR_CACHE, notKNNIndex);
        Request request = new Request(RestRequest.Method.POST.name(), restURI);

        ResponseException ex = expectThrows(ResponseException.class, () -> client().performRequest(request));
        assertTrue(ex.getMessage().contains(notKNNIndex));
    }

    public void testClearCacheSingleIndex() throws Exception {
        String testIndex = getTestName().toLowerCase();
        int graphCountBefore = getTotalGraphsInCache();
        createKnnIndex(testIndex, getKNNDefaultIndexSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
        addKnnDoc(testIndex, String.valueOf(randomInt()), TEST_FIELD, new Float[] { randomFloat(), randomFloat() });

        knnWarmup(Collections.singletonList(testIndex));

        assertEquals(graphCountBefore + 1, getTotalGraphsInCache());

        clearCache(Collections.singletonList(testIndex));
        assertEquals(graphCountBefore, getTotalGraphsInCache());
    }

    public void testClearCacheMultipleIndices() throws Exception {
        String testIndex1 = getTestName().toLowerCase();
        String testIndex2 = getTestName().toLowerCase() + 1;
        int graphCountBefore = getTotalGraphsInCache();

        createKnnIndex(testIndex1, getKNNDefaultIndexSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
        addKnnDoc(testIndex1, String.valueOf(randomInt()), TEST_FIELD, new Float[] { randomFloat(), randomFloat() });

        createKnnIndex(testIndex2, getKNNDefaultIndexSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
        addKnnDoc(testIndex2, String.valueOf(randomInt()), TEST_FIELD, new Float[] { randomFloat(), randomFloat() });

        knnWarmup(Arrays.asList(testIndex1, testIndex2));

        assertEquals(graphCountBefore + 2, getTotalGraphsInCache());

        clearCache(Arrays.asList(testIndex1, testIndex2));
        assertEquals(graphCountBefore, getTotalGraphsInCache());
    }
}
