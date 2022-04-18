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

public class WarmupTestsIT extends AbstractRollingUpgradeTestCase {
    private final String CLASS_NAME = "warmuptestsit";
    private final String testIndexName = KNN_BWC_PREFIX + CLASS_NAME + "test-index";
    private final String testFieldName = "test-field";
    private final int dimensions = 5;
    private final int k = 5;
    private final int addDocsCnt = 1;

    public void testKnnWarmup() throws Exception {
        waitForClusterHealthGreen("3");
        switch (getClusterType()) {
            case OLD:

                createKnnIndex(testIndexName, getKNNDefaultIndexSettings(), createKnnIndexMapping(testFieldName, dimensions));
                addKnnDocsBWCTests(testIndexName, testFieldName, dimensions, 0, addDocsCnt);
                break;
            case MIXED:

                if (isFirstMixedRound()) {
                    int graphCount = getTotalGraphsInCache();
                    knnWarmup(Collections.singletonList(testIndexName));
                    assertEquals(graphCount + addDocsCnt, getTotalGraphsInCache());

                    addKnnDocsBWCTests(testIndexName, testFieldName, dimensions, 10, addDocsCnt);

                    graphCount = getTotalGraphsInCache();
                    knnWarmup(Collections.singletonList(testIndexName));
                    assertEquals(graphCount + addDocsCnt, getTotalGraphsInCache());

                } else {
                    int graphCount = getTotalGraphsInCache();
                    knnWarmup(Collections.singletonList(testIndexName));
                    assertEquals(graphCount + (2 * addDocsCnt), getTotalGraphsInCache());

                    addKnnDocsBWCTests(testIndexName, testFieldName, dimensions, 20, addDocsCnt);

                    graphCount = getTotalGraphsInCache();
                    knnWarmup(Collections.singletonList(testIndexName));
                    assertEquals(graphCount + addDocsCnt, getTotalGraphsInCache());

                }

                break;
            case UPGRADED:

                int graphCount = getTotalGraphsInCache();
                knnWarmup(Collections.singletonList(testIndexName));
                assertEquals(graphCount + (3 * addDocsCnt), getTotalGraphsInCache());

                addKnnDocsBWCTests(testIndexName, testFieldName, dimensions, 30, addDocsCnt);

                graphCount = getTotalGraphsInCache();
                knnWarmup(Collections.singletonList(testIndexName));
                assertEquals(graphCount + addDocsCnt, getTotalGraphsInCache());
        }
    }

}
