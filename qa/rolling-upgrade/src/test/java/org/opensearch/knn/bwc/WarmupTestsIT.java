/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import java.util.Collections;

import static org.opensearch.knn.TestUtils.NODES_BWC_CLUSTER;

public class WarmupTestsIT extends AbstractRollingUpgradeTestCase {
    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSIONS = 5;
    private static int DOC_ID = 0;
    private static final int K = 5;
    private static final int NUM_DOCS = 10;
    private static int QUERY_COUNT = 0;

    public void testKnnWarmup() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        switch (getClusterType()) {
            case OLD:
                createKnnIndex(testIndex, getKNNDefaultIndexSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
                break;
            case MIXED:
                if (isFirstMixedRound()) {
                    int graphCountFirstRound = getTotalGraphsInCache();
                    knnWarmup(Collections.singletonList(testIndex));
                    assertTrue(getTotalGraphsInCache() > graphCountFirstRound);

                    DOC_ID = NUM_DOCS;
                    addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);

                    int graphCountFirstRoundUpdated = getTotalGraphsInCache();
                    knnWarmup(Collections.singletonList(testIndex));
                    assertTrue(getTotalGraphsInCache() > graphCountFirstRoundUpdated);

                    QUERY_COUNT = 2 * NUM_DOCS;
                    validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, QUERY_COUNT, K);
                } else {
                    int graphCountSecondRound = getTotalGraphsInCache();
                    knnWarmup(Collections.singletonList(testIndex));
                    assertTrue(getTotalGraphsInCache() > graphCountSecondRound);

                    DOC_ID = 2 * NUM_DOCS;
                    addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);

                    int graphCountSecondRoundUpdated = getTotalGraphsInCache();
                    knnWarmup(Collections.singletonList(testIndex));
                    assertTrue(getTotalGraphsInCache() > graphCountSecondRoundUpdated);

                    QUERY_COUNT = 3 * NUM_DOCS;
                    validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, QUERY_COUNT, K);
                }
                break;
            case UPGRADED:
                int graphCountUpgraded = getTotalGraphsInCache();
                knnWarmup(Collections.singletonList(testIndex));
                assertTrue(getTotalGraphsInCache() > graphCountUpgraded);

                DOC_ID = 3 * NUM_DOCS;
                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);

                int graphCountUpgradedUpdated = getTotalGraphsInCache();
                knnWarmup(Collections.singletonList(testIndex));
                assertTrue(getTotalGraphsInCache() > graphCountUpgradedUpdated);

                QUERY_COUNT = 4 * NUM_DOCS;
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, QUERY_COUNT, K);
                deleteKNNIndex(testIndex);
        }

    }

}
