/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.bwc;

import java.util.Collections;

import static org.opensearch.knn.TestUtils.KNN_BWC_PREFIX;

public class WarmupTestsIT extends AbstractRestartUpgradeTestCase {
    private final String CLASS_NAME = "warmuptestsit";
    private final String testIndexName = KNN_BWC_PREFIX + CLASS_NAME + "test-index";
    private final String testFieldName = "test-field";
    private final int dimensions = 5;
    private final int k = 1;
    private final int addDocsCnt = 3;

    public void testKnnWarmupDefaultIndexSettings() throws Exception {
        waitForClusterHealthGreen("3");

        if (isRunningAgainstOldCluster()) {
            knnWarmupDefaultIndexSettingsOldCluster();

        }
        else {
            knnWarmupDefaultIndexSettingsUpgradedCluster();
        }
    }

    public void knnWarmupDefaultIndexSettingsOldCluster() throws Exception{
        createKnnIndex(testIndexName, getKNNDefaultIndexSettings(), createKnnIndexMapping(testFieldName, dimensions));
        addKnnDocsBWCTests(testIndexName, testFieldName, dimensions, 0, addDocsCnt);
    }

    public void knnWarmupDefaultIndexSettingsUpgradedCluster() throws Exception{
        int graphCount = getTotalGraphsInCache();
        knnWarmup(Collections.singletonList(testIndexName));
        assertEquals(graphCount + 1, getTotalGraphsInCache());

        validateSearchBWCTests(testIndexName, testFieldName, dimensions, 3, k);

        addKnnDocsBWCTests(testIndexName, testFieldName, dimensions, 3, addDocsCnt);

        graphCount = getTotalGraphsInCache();
        knnWarmup(Collections.singletonList(testIndexName));
        assertEquals(graphCount + 1, getTotalGraphsInCache());

        validateSearchBWCTests(testIndexName, testFieldName, dimensions, 6, k);
    }
}
