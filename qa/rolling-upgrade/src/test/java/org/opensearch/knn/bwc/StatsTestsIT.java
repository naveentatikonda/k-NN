/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import org.apache.http.util.EntityUtils;
import org.opensearch.client.Response;
import org.opensearch.cluster.health.ClusterHealthStatus;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentType;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.opensearch.knn.TestUtils.INDEX_KNN;
import static org.opensearch.knn.TestUtils.NODES_BWC_CLUSTER;
import static org.opensearch.knn.TestUtils.NUMBER_OF_SHARDS;
import static org.opensearch.knn.TestUtils.NUMBER_OF_REPLICAS;
import static org.opensearch.knn.TestUtils.OPENSEARCH_KNN_MODELS_INDEX;
import static org.opensearch.knn.plugin.stats.StatNames.HIT_COUNT;
import static org.opensearch.knn.plugin.stats.StatNames.MISS_COUNT;
import static org.opensearch.knn.plugin.stats.StatNames.MODEL_INDEX_STATUS;

public class StatsTestsIT extends AbstractRollingUpgradeTestCase {
    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSIONS = 2;
    private static final int K = 1;
    private static final int ADD_DOCS_COUNT = 1;
    private static final String MODEL_INDEX_STATUS_NAME = MODEL_INDEX_STATUS.getName();

    // KNN Stats : Hit count, Miss count
    public void testKNNStatsHitMissCnt() throws Exception {
        switch (getClusterType()) {
            case OLD:
                createKnnIndex(testIndex, getKNNIndexSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, ADD_DOCS_COUNT);

                validateHitMissCnt(0, 0);
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, 1, K);
                validateHitMissCnt(0, 1);
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, 1, K);
                validateHitMissCnt(1, 1);
                break;
            case MIXED:
                if (isFirstMixedRound()) {
                    waitForClusterHealthGreen(NODES_BWC_CLUSTER);
                    validateHitMissCnt(0, 0);

                    // validateHitMissCnt(1, 0, 0);
                    // validateHitMissCnt(2, 0, 0);

                    addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 1, ADD_DOCS_COUNT);
                    validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, 2, K);
                    validateHitMissCnt(0, 2);

                    // validateHitMissCnt(1, 0, 0);
                    // validateHitMissCnt(2, 0, 0);

                    validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, 2, K);
                    validateHitMissCnt(2, 2);

                } else {
                    validateHitMissCnt(0, 0);
                    addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 2, ADD_DOCS_COUNT);
                    validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, 3, K);
                    validateHitMissCnt(0, 3);

                    validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, 3, K);
                    validateHitMissCnt(3, 3);
                }
                break;
            case UPGRADED:
                validateHitMissCnt(0, 0);
                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 3, ADD_DOCS_COUNT);
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, 4, K);
                validateHitMissCnt(0, 4);

                validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, 4, K);
                validateHitMissCnt(4, 4);
                deleteKNNIndex(testIndex);
        }

    }

    // KNN Stats : model_index_status
    public void testModelIndexHealthMetrics() throws IOException {
        switch (getClusterType()) {
            case OLD:
                Map<String, Object> statsMap = getModelIndexHealthMetric(MODEL_INDEX_STATUS_NAME);
                assertNull(statsMap.get(MODEL_INDEX_STATUS_NAME));

                createModelSystemIndex();
                break;
            case MIXED:
                if (isFirstMixedRound()) {
                    Map<String, Object> statsMapFirstMixedRound = getModelIndexHealthMetric(MODEL_INDEX_STATUS_NAME);
                    assertNotNull(statsMapFirstMixedRound.get(MODEL_INDEX_STATUS_NAME));
                    assertNotNull(ClusterHealthStatus.fromString((String) statsMapFirstMixedRound.get(MODEL_INDEX_STATUS_NAME)));
                } else {
                    deleteKNNIndex(OPENSEARCH_KNN_MODELS_INDEX);

                    Map<String, Object> statsMapSecondMixedRound = getModelIndexHealthMetric(MODEL_INDEX_STATUS_NAME);
                    assertNull(statsMapSecondMixedRound.get(MODEL_INDEX_STATUS_NAME));

                    createModelSystemIndex();
                }
                break;
            case UPGRADED:
                Map<String, Object> statsMapUpgraded = getModelIndexHealthMetric(MODEL_INDEX_STATUS_NAME);
                assertNotNull(statsMapUpgraded.get(MODEL_INDEX_STATUS_NAME));
                assertNotNull(ClusterHealthStatus.fromString((String) statsMapUpgraded.get(MODEL_INDEX_STATUS_NAME)));

                deleteKNNIndex(OPENSEARCH_KNN_MODELS_INDEX);
        }
    }

    public Map<String, Object> getModelIndexHealthMetric(String modelIndexStatusName) throws IOException {
        Response response = getKnnStats(Collections.emptyList(), Arrays.asList(modelIndexStatusName));
        String responseBody = EntityUtils.toString(response.getEntity());
        return createParser(XContentType.JSON.xContent(), responseBody).map();
    }

    public Settings getKNNIndexSettings() {
        return Settings.builder().put(NUMBER_OF_SHARDS, 3).put(NUMBER_OF_REPLICAS, 0).put(INDEX_KNN, true).build();
    }

    public void validateHitMissCnt(int expHitCount, int expMissCount) throws IOException {
        Response response = getKnnStats(Collections.emptyList(), Collections.emptyList());
        String responseBody = EntityUtils.toString(response.getEntity());
        List<Map<String, Object>> nodeStats = parseNodeStatsResponse(responseBody);

        int hitCount = nodeStats.stream().mapToInt(hitCntStats -> (int) hitCntStats.get(HIT_COUNT.getName())).sum();
        int missCount = nodeStats.stream().mapToInt(missCntStats -> (int) missCntStats.get(MISS_COUNT.getName())).sum();

        assertEquals(expHitCount, hitCount);
        assertEquals(expMissCount, missCount);
    }

}
