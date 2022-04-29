/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import org.opensearch.common.settings.Settings;
import org.opensearch.knn.index.SpaceType;

import java.util.Collections;
import java.util.Optional;

import static org.opensearch.knn.TestUtils.KNN_ALGO_PARAM_M_MIN_VALUE;
import static org.opensearch.knn.TestUtils.KNN_ALGO_PARAM_EF_CONSTRUCTION_MIN_VALUE;
import static org.opensearch.knn.TestUtils.KNN_ENGINE_FAISS;
import static org.opensearch.knn.TestUtils.NODES_BWC_CLUSTER;

public class WarmupTestsIT extends AbstractRestartUpgradeTestCase {
    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSIONS = 5;
    private static final int K = 5;
    private static final int ADD_DOCS_COUNT = 10;

    // Default Legacy Field Mapping
    // space_type : "l2", engine : "nmslib", m : 16, ef_construction : 512
    public void testKnnWarmupDefaultLegacyFieldMapping() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);

        if (isRunningAgainstOldCluster()) {
            createKnnIndex(testIndex, getKNNDefaultIndexSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS));
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, ADD_DOCS_COUNT);
        }

        else {
            kNNWarmupUpgradedCluster();
        }
    }

    // Custom Legacy Field Mapping
    // space_type : "linf", engine : "nmslib", m : 2, ef_construction : 2
    public void testKnnWarmupCustomLegacyFieldMapping() throws Exception {
        if (isRunningAgainstOldCluster()) {
            Settings indexMappingSettings = getKNNIndexCustomLegacyFieldMappingSettings(
                SpaceType.LINF,
                KNN_ALGO_PARAM_M_MIN_VALUE,
                KNN_ALGO_PARAM_EF_CONSTRUCTION_MIN_VALUE
            );
            String indexMapping = createKnnIndexMapping(TEST_FIELD, DIMENSIONS);
            createKnnIndex(testIndex, indexMappingSettings, indexMapping);
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, ADD_DOCS_COUNT);
        } else {
            kNNWarmupUpgradedCluster();
        }
    }

    // Default Method Field Mapping
    // space_type : "l2", engine : "nmslib", m : 16, ef_construction : 512
    public void testKnnWarmupDefaultMethodFieldMapping() throws Exception {
        if (isRunningAgainstOldCluster()) {
            createKnnIndex(testIndex, getKNNDefaultIndexSettings(), createKnnIndexMethodFieldMapping(TEST_FIELD, DIMENSIONS));
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, ADD_DOCS_COUNT);
        } else {
            kNNWarmupUpgradedCluster();
        }
    }

    // Custom Method Field Mapping
    // space_type : "innerproduct", engine : "faiss", m : 50, ef_construction : 1024
    public void testKnnWarmupCustomMethodFieldMapping() throws Exception {

        // Skip test if version is 1.0 or 1.1
        Optional<String> bwcVersion = getBWCVersion();
        if (bwcVersion.isEmpty() || bwcVersion.get().startsWith("1.0") || bwcVersion.get().startsWith("1.1")) {
            return;
        }
        if (isRunningAgainstOldCluster()) {
            createKnnIndex(
                testIndex,
                getKNNDefaultIndexSettings(),
                createKnnIndexCustomMethodFieldMapping(TEST_FIELD, DIMENSIONS, SpaceType.INNER_PRODUCT, KNN_ENGINE_FAISS, 50, 1024)
            );
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, ADD_DOCS_COUNT);
        } else {
            kNNWarmupUpgradedCluster();
        }
    }

    public void kNNWarmupUpgradedCluster() throws Exception {
        int graphCount = getTotalGraphsInCache();
        knnWarmup(Collections.singletonList(testIndex));
        assertTrue(getTotalGraphsInCache() > graphCount);

        validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, 10, K);

        addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 10, ADD_DOCS_COUNT);

        int updatedGraphCount = getTotalGraphsInCache();
        knnWarmup(Collections.singletonList(testIndex));
        assertTrue(getTotalGraphsInCache() > updatedGraphCount);

        validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, 20, K);
        deleteKNNIndex(testIndex);
    }
}
