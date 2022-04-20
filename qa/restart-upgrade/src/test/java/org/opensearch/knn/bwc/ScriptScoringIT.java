/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import org.opensearch.knn.index.SpaceType;

public class ScriptScoringIT extends AbstractRestartUpgradeTestCase {
    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSIONS = 5;
    private static final int K = 5;
    private static final int ADD_DOCS_CNT = 10;

    // KNN script scoring for space_type "l2"
    public void testKNNL2ScriptScore() throws Exception {
        if (isRunningAgainstOldCluster()) {
            createKnnIndex(testIndex, getKNNScriptScoreSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, ADD_DOCS_CNT);
        } else {
            validateKNNScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, ADD_DOCS_CNT, K, SpaceType.L2);
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 10, ADD_DOCS_CNT);
            validateKNNScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, (2 * ADD_DOCS_CNT), K, SpaceType.L2);
            deleteKNNIndex(testIndex);
        }
    }

    // KNN script scoring for space_type "l1"
    public void testKNNL1ScriptScore() throws Exception {
        if (isRunningAgainstOldCluster()) {
            createKnnIndex(testIndex, getKNNScriptScoreSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, ADD_DOCS_CNT);
        } else {
            validateKNNScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, ADD_DOCS_CNT, K, SpaceType.L1);
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 10, ADD_DOCS_CNT);
            validateKNNScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, (2 * ADD_DOCS_CNT), K, SpaceType.L1);
            deleteKNNIndex(testIndex);
        }
    }

    // KNN script scoring for space_type "linf"
    public void testKNNLinfScriptScore() throws Exception {
        if (isRunningAgainstOldCluster()) {
            createKnnIndex(testIndex, getKNNScriptScoreSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, ADD_DOCS_CNT);
        } else {
            validateKNNScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, ADD_DOCS_CNT, K, SpaceType.LINF);
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 10, ADD_DOCS_CNT);
            validateKNNScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, (2 * ADD_DOCS_CNT), K, SpaceType.LINF);
            deleteKNNIndex(testIndex);
        }
    }

    // KNN script scoring for space_type "innerproduct"
    // public void testKNNInnerProductScriptScore() throws Exception {
    // if(isRunningAgainstOldCluster()) {
    // createKnnIndex(testIndex, getKNNScriptScoreSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
    // addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, ADD_DOCS_CNT);
    // } else {
    // validateKNNScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, ADD_DOCS_CNT, K, SpaceType.INNER_PRODUCT);
    // addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 10, ADD_DOCS_CNT);
    // validateKNNScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, (2 * ADD_DOCS_CNT), K, SpaceType.INNER_PRODUCT);
    // deleteKNNIndex(testIndex);
    // }
    // }

    // KNN script scoring for space_type "cosinesimil"
    // public void testKNNCosinesimilScriptScore() throws Exception {
    // if(isRunningAgainstOldCluster()) {
    // createKnnIndex(testIndex, getKNNScriptScoreSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
    // addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, ADD_DOCS_CNT);
    // } else {
    // validateKNNScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, ADD_DOCS_CNT, K, SpaceType.COSINESIMIL);
    // addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 10, ADD_DOCS_CNT);
    // validateKNNScriptScoreSearch(testIndex, TEST_FIELD, DIMENSIONS, (2 * ADD_DOCS_CNT), K, SpaceType.COSINESIMIL);
    // deleteKNNIndex(testIndex);
    // }
    // }

}
