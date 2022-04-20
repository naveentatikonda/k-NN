/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

public class ScriptScoringIT extends AbstractRollingUpgradeTestCase {
    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSIONS = 5;
    private static final int K = 5;
    private static final int ADD_DOCS_CNT = 10;

    // KNN script scoring for space_type "l2"
    public void testKNNL2ScriptScore() throws Exception {
        switch (getClusterType()) {
            case OLD:
                createKnnIndex(testIndex, getKNNScriptScoreSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, ADD_DOCS_CNT);
                break;
            case MIXED:
                if (isFirstMixedRound()) {
                    validateKNNL2ScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, ADD_DOCS_CNT, K);
                    addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 10, ADD_DOCS_CNT);
                    validateKNNL2ScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, (2 * ADD_DOCS_CNT), K);
                } else {
                    validateKNNL2ScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, (2 * ADD_DOCS_CNT), K);
                    addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 20, ADD_DOCS_CNT);
                    validateKNNL2ScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, (3 * ADD_DOCS_CNT), K);
                }
                break;
            case UPGRADED:
                validateKNNL2ScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, (3 * ADD_DOCS_CNT), K);
                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 30, ADD_DOCS_CNT);
                validateKNNL2ScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, (4 * ADD_DOCS_CNT), K);
                deleteKNNIndex(testIndex);
        }
    }

}
