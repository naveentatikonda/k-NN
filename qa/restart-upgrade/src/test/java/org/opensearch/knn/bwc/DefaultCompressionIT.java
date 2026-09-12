/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import org.opensearch.Version;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

import java.util.Map;
import java.util.Optional;

import static org.opensearch.knn.TestUtils.KNN_VECTOR;
import static org.opensearch.knn.TestUtils.NODES_BWC_CLUSTER;
import static org.opensearch.knn.TestUtils.PROPERTIES;
import static org.opensearch.knn.TestUtils.VECTOR_TYPE;
import static org.opensearch.knn.common.KNNConstants.COMPRESSION_LEVEL_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.MODE_PARAMETER;

public class DefaultCompressionIT extends AbstractRestartUpgradeTestCase {
    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSIONS = 64;
    private static final int K = 5;
    private static final int NUM_DOCS = 10;

    @SuppressWarnings("unchecked")
    public void testRestartUpgrade_defaultCompression() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        final String explicitX32Index = testIndex + "-explicit-x32";
        final String explicitX16Index = testIndex + "-explicit-x16";
        final String explicitX8Index = testIndex + "-explicit-x8";
        final boolean compressionSupported = isCompressionSupported(getBWCVersion());

        if (isRunningAgainstOldCluster()) {
            if (compressionSupported) {
                createExplicitCompressionIndex(explicitX32Index, CompressionLevel.x32);
                createExplicitCompressionIndex(explicitX16Index, CompressionLevel.x16);
                createExplicitCompressionIndex(explicitX8Index, CompressionLevel.x8);
            }

            createKnnIndex(testIndex, getKNNDefaultIndexSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS, METHOD_HNSW, FAISS_NAME));
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, NUM_DOCS);
            flush(testIndex, true);
        } else {
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);

            Map<String, Object> defaultMappings = getIndexMappingAsMap(testIndex);
            Map<String, Object> properties = (Map<String, Object>) defaultMappings.get(PROPERTIES);
            assertNotNull("Properties should not be null", properties);
            Map<String, Object> defaultFieldProps = (Map<String, Object>) properties.get(TEST_FIELD);
            assertNotNull("Field properties should not be null", defaultFieldProps);
            assertNull(defaultFieldProps.get(COMPRESSION_LEVEL_PARAMETER));

            if (compressionSupported) {
                validateExplicitCompressionIndex(explicitX32Index, CompressionLevel.x32);
                validateExplicitCompressionIndex(explicitX16Index, CompressionLevel.x16);
                validateExplicitCompressionIndex(explicitX8Index, CompressionLevel.x8);
            }

            deleteKNNIndex(testIndex);
        }
    }

    private void createExplicitCompressionIndex(String indexName, CompressionLevel compressionLevel) throws Exception {
        String mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES)
            .startObject(TEST_FIELD)
            .field(VECTOR_TYPE, KNN_VECTOR)
            .field(DIMENSION, DIMENSIONS)
            .field(COMPRESSION_LEVEL_PARAMETER, compressionLevel.getName())
            .field(MODE_PARAMETER, Mode.ON_DISK.getName())
            .endObject()
            .endObject()
            .endObject()
            .toString();
        createKnnIndex(indexName, getKNNDefaultIndexSettings(), mapping);
        addKNNDocs(indexName, TEST_FIELD, DIMENSIONS, 0, NUM_DOCS);
        flush(indexName, true);
    }

    @SuppressWarnings("unchecked")
    private void validateExplicitCompressionIndex(String indexName, CompressionLevel compressionLevel) throws Exception {
        validateKNNSearch(indexName, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);

        Map<String, Object> mappings = getIndexMappingAsMap(indexName);
        Map<String, Object> properties = (Map<String, Object>) mappings.get(PROPERTIES);
        assertNotNull(compressionLevel.getName() + " properties should not be null", properties);
        Map<String, Object> fieldProps = (Map<String, Object>) properties.get(TEST_FIELD);
        assertNotNull(compressionLevel.getName() + " field properties should not be null", fieldProps);
        assertEquals(compressionLevel.getName(), fieldProps.get(COMPRESSION_LEVEL_PARAMETER));
        assertEquals(Mode.ON_DISK.getName(), fieldProps.get(MODE_PARAMETER));
        deleteKNNIndex(indexName);
    }

    private boolean isCompressionSupported(final Optional<String> bwcVersion) {
        if (bwcVersion.isEmpty()) {
            return false;
        }
        String versionString = bwcVersion.get().replace("-SNAPSHOT", "");
        return Version.fromString(versionString).onOrAfter(Version.V_2_17_0);
    }

}
