/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import static org.opensearch.knn.TestUtils.NODES_BWC_CLUSTER;

public class IndexingIT extends AbstractRestartUpgradeTestCase {
    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSIONS = 5;
    private static final int K = 5;
    private static final int ADD_DOCS_CNT = 10;

    public void testKnnDefaultIndexSettings() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);

        if (isRunningAgainstOldCluster()) {
            createKnnIndex(testIndex, getKNNDefaultIndexSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
            addKnnDocsBWCTests(testIndex, TEST_FIELD, DIMENSIONS, 0, ADD_DOCS_CNT);
        }

        else {
            validateSearchBWCTests(testIndex, TEST_FIELD, DIMENSIONS, 10, K);
            cleanUpCache();
            addKnnDocsBWCTests(testIndex, TEST_FIELD, DIMENSIONS, 10, ADD_DOCS_CNT);
            validateSearchBWCTests(testIndex, TEST_FIELD, DIMENSIONS, 20, K);
            forceMergeKnnIndex(testIndex);
            validateSearchBWCTests(testIndex, TEST_FIELD, DIMENSIONS, 20, K);
            deleteKNNIndex(testIndex);
        }
    }

}
