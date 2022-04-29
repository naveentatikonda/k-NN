/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

public class PainlessScriptScoringIT extends AbstractRollingUpgradeTestCase {
    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSIONS = 5;
    private static final int K = 5;
    private static final int ADD_DOCS_COUNT = 10;

    // KNN painless script scoring for space_type "l2"
    public void testKNNL2PainlessScriptScore() throws Exception {
        switch (getClusterType()) {
            case OLD:
                createKnnIndex(testIndex, getKNNScriptScoreSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, ADD_DOCS_COUNT);
                break;
            case MIXED:
                if (isFirstMixedRound()) {
                    String source = generateL2PainlessScriptSource(TEST_FIELD, DIMENSIONS, ADD_DOCS_COUNT);
                    validateKNNPainlessScriptScoreSearch(testIndex, TEST_FIELD, source, ADD_DOCS_COUNT, K);

                    addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 10, ADD_DOCS_COUNT);

                    source = generateL2PainlessScriptSource(TEST_FIELD, DIMENSIONS, 2 * ADD_DOCS_COUNT);
                    validateKNNPainlessScriptScoreSearch(testIndex, TEST_FIELD, source, 2 * ADD_DOCS_COUNT, K);
                } else {
                    String source = generateL2PainlessScriptSource(TEST_FIELD, DIMENSIONS, 2 * ADD_DOCS_COUNT);
                    validateKNNPainlessScriptScoreSearch(testIndex, TEST_FIELD, source, 2 * ADD_DOCS_COUNT, K);

                    addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 20, ADD_DOCS_COUNT);

                    source = generateL2PainlessScriptSource(TEST_FIELD, DIMENSIONS, 3 * ADD_DOCS_COUNT);
                    validateKNNPainlessScriptScoreSearch(testIndex, TEST_FIELD, source, 3 * ADD_DOCS_COUNT, K);
                }
                break;
            case UPGRADED:
                String source = generateL2PainlessScriptSource(TEST_FIELD, DIMENSIONS, 3 * ADD_DOCS_COUNT);
                validateKNNPainlessScriptScoreSearch(testIndex, TEST_FIELD, source, 3 * ADD_DOCS_COUNT, K);

                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 30, ADD_DOCS_COUNT);

                source = generateL2PainlessScriptSource(TEST_FIELD, DIMENSIONS, 4 * ADD_DOCS_COUNT);
                validateKNNPainlessScriptScoreSearch(testIndex, TEST_FIELD, source, 4 * ADD_DOCS_COUNT, K);
                deleteKNNIndex(testIndex);
        }
    }

    // KNN painless script scoring for space_type "l1"
    public void testKNNL1PainlessScriptScore() throws Exception {
        switch (getClusterType()) {
            case OLD:
                createKnnIndex(testIndex, getKNNScriptScoreSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, ADD_DOCS_COUNT);
                break;
            case MIXED:
                if (isFirstMixedRound()) {
                    String source = generateL1PainlessScriptSource(TEST_FIELD, DIMENSIONS, ADD_DOCS_COUNT);
                    validateKNNPainlessScriptScoreSearch(testIndex, TEST_FIELD, source, ADD_DOCS_COUNT, K);

                    addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 10, ADD_DOCS_COUNT);

                    source = generateL1PainlessScriptSource(TEST_FIELD, DIMENSIONS, 2 * ADD_DOCS_COUNT);
                    validateKNNPainlessScriptScoreSearch(testIndex, TEST_FIELD, source, 2 * ADD_DOCS_COUNT, K);
                } else {
                    String source = generateL1PainlessScriptSource(TEST_FIELD, DIMENSIONS, 2 * ADD_DOCS_COUNT);
                    validateKNNPainlessScriptScoreSearch(testIndex, TEST_FIELD, source, 2 * ADD_DOCS_COUNT, K);

                    addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 20, ADD_DOCS_COUNT);

                    source = generateL1PainlessScriptSource(TEST_FIELD, DIMENSIONS, 3 * ADD_DOCS_COUNT);
                    validateKNNPainlessScriptScoreSearch(testIndex, TEST_FIELD, source, 3 * ADD_DOCS_COUNT, K);
                }
                break;
            case UPGRADED:
                String source = generateL1PainlessScriptSource(TEST_FIELD, DIMENSIONS, 3 * ADD_DOCS_COUNT);
                validateKNNPainlessScriptScoreSearch(testIndex, TEST_FIELD, source, 3 * ADD_DOCS_COUNT, K);

                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 30, ADD_DOCS_COUNT);

                source = generateL1PainlessScriptSource(TEST_FIELD, DIMENSIONS, 4 * ADD_DOCS_COUNT);
                validateKNNPainlessScriptScoreSearch(testIndex, TEST_FIELD, source, 4 * ADD_DOCS_COUNT, K);
                deleteKNNIndex(testIndex);
        }
    }

}
