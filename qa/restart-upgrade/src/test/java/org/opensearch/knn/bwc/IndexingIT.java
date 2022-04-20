/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import org.opensearch.knn.index.SpaceType;

import static org.opensearch.knn.TestUtils.*;

public class IndexingIT extends AbstractRestartUpgradeTestCase {
    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSIONS = 5;
    private static final int K = 5;
    private static final int ADD_DOCS_CNT = 10;

    // Default Legacy Field Mapping
    // space_type : "l2", engine : "nmslib", m : 16, ef_construction : 512
    public void testKnnIndexDefaultLegacyFieldMapping() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);

        if (isRunningAgainstOldCluster()) {
            createKnnIndex(testIndex, getKNNDefaultIndexSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, ADD_DOCS_CNT);
        }

        else {
            kNNIndexUpgradedCluster();
        }
    }

    // Custom Legacy Field Mapping
    // space_type : "linf", engine : "nmslib", m : 2, ef_construction : 2
    public void testKnnIndexCustomLegacyFieldMapping() throws Exception {
        if (isRunningAgainstOldCluster()) {
            createKnnIndex(
                testIndex,
                getKNNIndexCustomLegacyFieldMappingSettings(
                    SpaceType.LINF,
                    KNN_ALGO_PARAM_M_MIN_VALUE,
                    KNN_ALGO_PARAM_EF_CONSTRUCTION_MIN_VALUE
                ),
                createKnnIndexMapping(TEST_FIELD, DIMENSIONS)
            );
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, ADD_DOCS_CNT);
        } else {
            kNNIndexUpgradedCluster();
        }
    }

    // Default Method Field Mapping
    // space_type : "l2", engine : "nmslib", m : 16, ef_construction : 512
    public void testKnnIndexDefaultMethodFieldMapping() throws Exception {
        if (isRunningAgainstOldCluster()) {
            createKnnIndex(testIndex, getKNNDefaultIndexSettings(), createKnnIndexMethodFieldMapping(TEST_FIELD, DIMENSIONS));
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, ADD_DOCS_CNT);
        } else {
            kNNIndexUpgradedCluster();
        }
    }

    // Custom Method Field Mapping
    // space_type : "l2", engine : "faiss", m : 2, ef_construction : 2
    public void testKnnIndexCustomMethodFieldMapping() throws Exception {
        if (isRunningAgainstOldCluster()) {
            createKnnIndex(
                testIndex,
                getKNNDefaultIndexSettings(),
                createKnnIndexCustomMethodFieldMapping(
                    TEST_FIELD,
                    DIMENSIONS,
                    SpaceType.INNER_PRODUCT,
                    KNN_ENGINE_FAISS,
                    KNN_ALGO_PARAM_M_MIN_VALUE,
                    KNN_ALGO_PARAM_EF_CONSTRUCTION_MIN_VALUE
                )
            );
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, ADD_DOCS_CNT);
        } else {
            kNNIndexUpgradedCluster();
        }
    }

    public void kNNIndexUpgradedCluster() throws Exception {
        validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, 10, K);
        cleanUpCache();
        addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 10, ADD_DOCS_CNT);
        validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, 20, K);
        forceMergeKnnIndex(testIndex);
        validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, 20, K);
        deleteKNNIndex(testIndex);

    }
}
