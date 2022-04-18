/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import static org.opensearch.knn.TestUtils.KNN_BWC_PREFIX;

public class IndexingIT extends AbstractRestartUpgradeTestCase {
    private final String CLASS_NAME = "indexingit";
    private final String testIndexName = KNN_BWC_PREFIX + CLASS_NAME + "test-index";
    private final String testFieldName = "test-field";
    private final int dimensions = 5;
    private final int k = 5;
    private final int addDocsCnt = 10;

    public void testKnnDefaultIndexSettings() throws Exception {
        waitForClusterHealthGreen("3");

        if (isRunningAgainstOldCluster()) {
            createKnnIndex(testIndexName, getKNNDefaultIndexSettings(), createKnnIndexMapping(testFieldName, dimensions));
            addKnnDocsBWCTests(testIndexName, testFieldName, dimensions, 0, addDocsCnt);
        }

        else {
            validateSearchBWCTests(testIndexName, testFieldName, dimensions, 10, k);
            cleanUpCache();
            addKnnDocsBWCTests(testIndexName, testFieldName, dimensions, 10, addDocsCnt);
            validateSearchBWCTests(testIndexName, testFieldName, dimensions, 20, k);
            forceMergeKnnIndex(testIndexName);
            validateSearchBWCTests(testIndexName, testFieldName, dimensions, 20, k);
        }
    }
}
