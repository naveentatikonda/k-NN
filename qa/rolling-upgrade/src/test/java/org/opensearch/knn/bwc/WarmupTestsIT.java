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
    private static final int K = 5;
    private static final int ADD_DOCS_COUNT = 10;

    public void testKnnWarmup() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        switch (getClusterType()) {
            case OLD:
                createKnnIndex(testIndex, getKNNDefaultIndexSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, ADD_DOCS_COUNT);
                break;
            case MIXED:
                if (isFirstMixedRound()) {
                    int graphCountFirstRound = getTotalGraphsInCache();
                    knnWarmup(Collections.singletonList(testIndex));
                    assertTrue(getTotalGraphsInCache() > graphCountFirstRound);

                    addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 10, ADD_DOCS_COUNT);

                    int graphCountFirstRoundUpdated = getTotalGraphsInCache();
                    knnWarmup(Collections.singletonList(testIndex));
                    assertTrue(getTotalGraphsInCache() > graphCountFirstRoundUpdated);

                    validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, 20, K);
                } else {
                    int graphCountSecondRound = getTotalGraphsInCache();
                    knnWarmup(Collections.singletonList(testIndex));
                    assertTrue(getTotalGraphsInCache() > graphCountSecondRound);

                    addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 20, ADD_DOCS_COUNT);

                    int graphCountSecondRoundUpdated = getTotalGraphsInCache();
                    knnWarmup(Collections.singletonList(testIndex));
                    assertTrue(getTotalGraphsInCache() > graphCountSecondRoundUpdated);

                    validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, 30, K);
                }
                break;
            case UPGRADED:
                int graphCountUpgraded = getTotalGraphsInCache();
                knnWarmup(Collections.singletonList(testIndex));
                assertTrue(getTotalGraphsInCache() > graphCountUpgraded);

                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 30, ADD_DOCS_COUNT);

                int graphCountUpgradedUpdated = getTotalGraphsInCache();
                knnWarmup(Collections.singletonList(testIndex));
                assertTrue(getTotalGraphsInCache() > graphCountUpgradedUpdated);

                validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, 40, K);
                deleteKNNIndex(testIndex);
        }

    }

}
