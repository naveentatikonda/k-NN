/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

public class PainlessScriptScoringIT extends AbstractRollingUpgradeTestCase {
    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSIONS = 5;
    private static int DOC_ID = 0;
    private static final int K = 5;
    private static final int NUM_DOCS = 10;
    private static int QUERY_COUNT = 0;

    // KNN painless script scoring for space_type "l2"
    public void testKNNL2PainlessScriptScore() throws Exception {
        switch (getClusterType()) {
            case OLD:
                createKnnIndex(testIndex, getKNNScriptScoreSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
                break;
            case MIXED:
                if (isFirstMixedRound()) {
                    DOC_ID = NUM_DOCS;
                    QUERY_COUNT = NUM_DOCS;
                    String source = generateL2PainlessScriptSource(TEST_FIELD, DIMENSIONS, QUERY_COUNT);
                    validateKNNPainlessScriptScoreSearch(testIndex, TEST_FIELD, source, QUERY_COUNT, K);

                    addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);

                    QUERY_COUNT = 2 * NUM_DOCS;
                    source = generateL2PainlessScriptSource(TEST_FIELD, DIMENSIONS, QUERY_COUNT);
                    validateKNNPainlessScriptScoreSearch(testIndex, TEST_FIELD, source, QUERY_COUNT, K);
                } else {
                    DOC_ID = 2 * NUM_DOCS;
                    QUERY_COUNT = 2 * NUM_DOCS;
                    String source = generateL2PainlessScriptSource(TEST_FIELD, DIMENSIONS, QUERY_COUNT);
                    validateKNNPainlessScriptScoreSearch(testIndex, TEST_FIELD, source, QUERY_COUNT, K);

                    addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);

                    QUERY_COUNT = 3 * NUM_DOCS;
                    source = generateL2PainlessScriptSource(TEST_FIELD, DIMENSIONS, QUERY_COUNT);
                    validateKNNPainlessScriptScoreSearch(testIndex, TEST_FIELD, source, QUERY_COUNT, K);
                }
                break;
            case UPGRADED:
                DOC_ID = 3 * NUM_DOCS;
                QUERY_COUNT = 3 * NUM_DOCS;
                String source = generateL2PainlessScriptSource(TEST_FIELD, DIMENSIONS, QUERY_COUNT);
                validateKNNPainlessScriptScoreSearch(testIndex, TEST_FIELD, source, QUERY_COUNT, K);

                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);

                QUERY_COUNT = 4 * NUM_DOCS;
                source = generateL2PainlessScriptSource(TEST_FIELD, DIMENSIONS, QUERY_COUNT);
                validateKNNPainlessScriptScoreSearch(testIndex, TEST_FIELD, source, QUERY_COUNT, K);
                deleteKNNIndex(testIndex);
        }
    }

    // KNN painless script scoring for space_type "l1"
    public void testKNNL1PainlessScriptScore() throws Exception {
        switch (getClusterType()) {
            case OLD:
                createKnnIndex(testIndex, getKNNScriptScoreSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);
                break;
            case MIXED:
                if (isFirstMixedRound()) {
                    DOC_ID = NUM_DOCS;
                    QUERY_COUNT = NUM_DOCS;
                    String source = generateL1PainlessScriptSource(TEST_FIELD, DIMENSIONS, QUERY_COUNT);
                    validateKNNPainlessScriptScoreSearch(testIndex, TEST_FIELD, source, QUERY_COUNT, K);

                    addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);

                    QUERY_COUNT = 2 * NUM_DOCS;
                    source = generateL1PainlessScriptSource(TEST_FIELD, DIMENSIONS, QUERY_COUNT);
                    validateKNNPainlessScriptScoreSearch(testIndex, TEST_FIELD, source, QUERY_COUNT, K);
                } else {
                    DOC_ID = 2 * NUM_DOCS;
                    QUERY_COUNT = 2 * NUM_DOCS;
                    String source = generateL1PainlessScriptSource(TEST_FIELD, DIMENSIONS, QUERY_COUNT);
                    validateKNNPainlessScriptScoreSearch(testIndex, TEST_FIELD, source, QUERY_COUNT, K);

                    addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);

                    QUERY_COUNT = 3 * NUM_DOCS;
                    source = generateL1PainlessScriptSource(TEST_FIELD, DIMENSIONS, QUERY_COUNT);
                    validateKNNPainlessScriptScoreSearch(testIndex, TEST_FIELD, source, QUERY_COUNT, K);
                }
                break;
            case UPGRADED:
                DOC_ID = 3 * NUM_DOCS;
                QUERY_COUNT = 3 * NUM_DOCS;
                String source = generateL1PainlessScriptSource(TEST_FIELD, DIMENSIONS, QUERY_COUNT);
                validateKNNPainlessScriptScoreSearch(testIndex, TEST_FIELD, source, QUERY_COUNT, K);

                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, DOC_ID, NUM_DOCS);

                QUERY_COUNT = 4 * NUM_DOCS;
                source = generateL1PainlessScriptSource(TEST_FIELD, DIMENSIONS, QUERY_COUNT);
                validateKNNPainlessScriptScoreSearch(testIndex, TEST_FIELD, source, QUERY_COUNT, K);
                deleteKNNIndex(testIndex);
        }
    }

}
