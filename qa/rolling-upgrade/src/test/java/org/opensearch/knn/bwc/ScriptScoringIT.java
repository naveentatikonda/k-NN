/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import org.opensearch.knn.index.SpaceType;

public class ScriptScoringIT extends AbstractRollingUpgradeTestCase {
    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSIONS = 5;
    private static final int K = 5;
    private static final int ADD_DOCS_COUNT = 10;

    // KNN script scoring for space_type "l2"
    public void testKNNL2ScriptScore() throws Exception {
        switch (getClusterType()) {
            case OLD:
                createKnnIndex(testIndex, getKNNScriptScoreSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, ADD_DOCS_COUNT);
                break;
            case MIXED:
                if (isFirstMixedRound()) {
                    validateKNNScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, ADD_DOCS_COUNT, K, SpaceType.L2);
                    addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 10, ADD_DOCS_COUNT);
                    validateKNNScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, (2 * ADD_DOCS_COUNT), K, SpaceType.L2);
                } else {
                    validateKNNScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, (2 * ADD_DOCS_COUNT), K, SpaceType.L2);
                    addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 20, ADD_DOCS_COUNT);
                    validateKNNScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, (3 * ADD_DOCS_COUNT), K, SpaceType.L2);
                }
                break;
            case UPGRADED:
                validateKNNScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, (3 * ADD_DOCS_COUNT), K, SpaceType.L2);
                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 30, ADD_DOCS_COUNT);
                validateKNNScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, (4 * ADD_DOCS_COUNT), K, SpaceType.L2);
                deleteKNNIndex(testIndex);
        }
    }

}
