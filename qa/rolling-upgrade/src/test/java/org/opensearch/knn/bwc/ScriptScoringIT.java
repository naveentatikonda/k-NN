/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import org.opensearch.knn.index.SpaceType;

public class ScriptScoringIT extends AbstractRollingUpgradeTestCase {
    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSIONS = 5;
    private static int DOC_ID = 0;
    private static final int K = 5;
    private static final int NUM_DOCS = 10;
    private static int QUERY_COUNT = 0;

    // KNN script scoring for space_type "l2"
    public void testKNNL2ScriptScore() throws Exception {
        switch (getClusterType()) {
            case OLD:
                createKnnIndex(testIndex, getKNNScriptScoreSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
                break;
            case MIXED:
                if (isFirstMixedRound()) {
                    QUERY_COUNT = NUM_DOCS;
                    DOC_ID = NUM_DOCS;
                    validateKNNScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, QUERY_COUNT, K, SpaceType.L2);
                    addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
                    QUERY_COUNT = 2 * NUM_DOCS;
                    validateKNNScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, QUERY_COUNT, K, SpaceType.L2);
                } else {
                    QUERY_COUNT = 2 * NUM_DOCS;
                    DOC_ID = 2 * NUM_DOCS;
                    validateKNNScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, QUERY_COUNT, K, SpaceType.L2);
                    addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
                    QUERY_COUNT = 3 * NUM_DOCS;
                    validateKNNScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, QUERY_COUNT, K, SpaceType.L2);
                }
                break;
            case UPGRADED:
                QUERY_COUNT = 3 * NUM_DOCS;
                DOC_ID = 3 * NUM_DOCS;
                validateKNNScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, QUERY_COUNT, K, SpaceType.L2);
                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
                QUERY_COUNT = 4 * NUM_DOCS;
                validateKNNScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, QUERY_COUNT, K, SpaceType.L2);
                deleteKNNIndex(testIndex);
        }
    }

}
