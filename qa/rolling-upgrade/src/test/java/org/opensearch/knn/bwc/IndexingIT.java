/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import static org.opensearch.knn.TestUtils.*;

public class IndexingIT extends AbstractRollingUpgradeTestCase {
    private final String CLASS_NAME = "indexingit";
    private final String testIndexName = KNN_BWC_PREFIX + CLASS_NAME + "test-index";
    private final String testFieldName = "test-field";
    private final int dimensions = 5;
    private final int k = 5;
    private final int addDocsCnt = 10;

    public void testKnnDefaultIndexSettings() throws Exception {
        waitForClusterHealthGreen("3");
        switch (getClusterType()) {
            case OLD:

                createKnnIndex(testIndexName, getKNNDefaultIndexSettings(), createKnnIndexMapping(testFieldName, dimensions));
                addKnnDocsBWCTests(testIndexName, testFieldName, dimensions, 0, addDocsCnt);
                break;
            case MIXED:

                if (isFirstMixedRound()) {
                    validateSearchBWCTests(testIndexName, testFieldName, dimensions, 10, k);
                    addKnnDocsBWCTests(testIndexName, testFieldName, dimensions, 10, addDocsCnt);
                    validateSearchBWCTests(testIndexName, testFieldName, dimensions, 20, k);

                } else {
                    validateSearchBWCTests(testIndexName, testFieldName, dimensions, 20, k);
                    addKnnDocsBWCTests(testIndexName, testFieldName, dimensions, 20, addDocsCnt);
                    validateSearchBWCTests(testIndexName, testFieldName, dimensions, 30, k);
                }

                break;
            case UPGRADED:

                validateSearchBWCTests(testIndexName, testFieldName, dimensions, 30, k);
                addKnnDocsBWCTests(testIndexName, testFieldName, dimensions, 30, addDocsCnt);
                validateSearchBWCTests(testIndexName, testFieldName, dimensions, 40, k);
                forceMergeKnnIndex(testIndexName);
                validateSearchBWCTests(testIndexName, testFieldName, dimensions, 40, k);
                break;
        }
    }
}
