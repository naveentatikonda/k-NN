/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import org.apache.http.util.EntityUtils;
import org.opensearch.client.Response;
import org.opensearch.knn.plugin.stats.StatNames;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class StatsTestsIT extends AbstractRollingUpgradeTestCase {
    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSIONS = 2;
    private static final int K = 1;
    private static final int ADD_DOCS_CNT = 1;

    // KNN Stats : Hit count, Miss count
    // public void testKNNStatsHitMissCnt() throws Exception {
    // switch (getClusterType()) {
    // case OLD:
    // createKnnIndex(testIndex, getKNNDefaultIndexSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
    // addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, ADD_DOCS_CNT);
    //
    // validateHitMissCnt(0, 0, 0);
    // validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, 1, K);
    // validateHitMissCnt(0, 0, 1);
    // validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, 1, K);
    // validateHitMissCnt(0, 1, 1);
    //
    // //validateHitMissCnt(1, 0, 0);
    // //validateHitMissCnt(2, 0, 0);
    // break;
    // case MIXED:
    // if(isFirstMixedRound()) {
    // waitForClusterHealthGreen(NODES_BWC_CLUSTER);
    // validateHitMissCnt(0, 0, 0);
    //
    // //validateHitMissCnt(1, 0, 0);
    // //validateHitMissCnt(2, 0, 0);
    //
    // addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 1, ADD_DOCS_CNT);
    // validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, 2, K);
    // validateHitMissCnt(0, 0, 2);
    //
    // //validateHitMissCnt(1, 0, 0);
    // //validateHitMissCnt(2, 0, 0);
    //
    // validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, 2, K);
    // validateHitMissCnt(0, 2, 2);
    //
    //
    // }
    // else {
    // validateHitMissCnt(0, 0, 0);
    // addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 2, ADD_DOCS_CNT);
    // validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, 3, K);
    // validateHitMissCnt(0, 0, 3);
    //
    // validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, 3, K);
    // validateHitMissCnt(0, 3, 3);
    // }
    // break;
    // case UPGRADED:
    // validateHitMissCnt(0, 0, 0);
    // addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 3, ADD_DOCS_CNT);
    // validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, 4, K);
    // validateHitMissCnt(0, 0, 4);
    //
    // validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, 4, K);
    // validateHitMissCnt(0, 4, 4);
    // deleteKNNIndex(testIndex);
    // }
    // }

    public void validateHitMissCnt(int node, int expHitCnt, int expMissCnt) throws IOException {
        Map<String, Object> nodeStats = getKnnStatsAllNodes().get(node);
        int hitCount = (int) nodeStats.get(StatNames.HIT_COUNT.getName());
        int missCount = (int) nodeStats.get(StatNames.MISS_COUNT.getName());
        assertEquals(expHitCnt, hitCount);
        assertEquals(expMissCnt, missCount);
    }

    public List<Map<String, Object>> getKnnStatsAllNodes() throws IOException {
        Response response = getKnnStats(Collections.emptyList(), Collections.emptyList());
        String responseBody = EntityUtils.toString(response.getEntity());

        return parseNodeStatsResponse(responseBody);
    }

}
