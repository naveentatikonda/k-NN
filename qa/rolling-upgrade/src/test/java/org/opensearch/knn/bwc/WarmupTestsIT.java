/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

public class WarmupTestsIT extends AbstractRollingUpgradeTestCase {
    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSIONS = 5;
    private static final int K = 5;
    private static final int ADD_DOCS_CNT = 10;

    // public void testKnnWarmup() throws Exception {
    // waitForClusterHealthGreen(NODES_BWC_CLUSTER);
    // switch (getClusterType()) {
    // case OLD:
    // createKnnIndex(testIndex, getKNNDefaultIndexSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
    // addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, ADD_DOCS_CNT);
    // break;
    // case MIXED:
    // if (isFirstMixedRound()) {
    // int graphCount = getTotalGraphsInCache();
    // knnWarmup(Collections.singletonList(testIndex));
    // assertEquals(graphCount + 1, getTotalGraphsInCache());
    //
    // addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 10, ADD_DOCS_CNT);
    //
    // graphCount = getTotalGraphsInCache();
    // knnWarmup(Collections.singletonList(testIndex));
    // assertEquals(graphCount + ADD_DOCS_CNT, getTotalGraphsInCache());
    //
    // validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, 20, K);
    // } else {
    // int graphCount = getTotalGraphsInCache();
    // knnWarmup(Collections.singletonList(testIndex));
    // assertEquals(graphCount + 2, getTotalGraphsInCache());
    //
    // addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 20, ADD_DOCS_CNT);
    //
    // graphCount = getTotalGraphsInCache();
    // knnWarmup(Collections.singletonList(testIndex));
    // assertEquals(graphCount + ADD_DOCS_CNT, getTotalGraphsInCache());
    //
    // validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, 30, K);
    // }
    // break;
    // case UPGRADED:
    // int graphCount = getTotalGraphsInCache();
    // knnWarmup(Collections.singletonList(testIndex));
    // assertEquals(graphCount + 3, getTotalGraphsInCache());
    //
    // addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 30, ADD_DOCS_CNT);
    //
    // graphCount = getTotalGraphsInCache();
    // knnWarmup(Collections.singletonList(testIndex));
    // assertEquals(graphCount + ADD_DOCS_CNT, getTotalGraphsInCache());
    //
    // validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, 40, K);
    // deleteKNNIndex(testIndex);
    // }
    // }

}
