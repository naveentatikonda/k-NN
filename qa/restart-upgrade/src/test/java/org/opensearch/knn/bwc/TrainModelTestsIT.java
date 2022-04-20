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

import org.apache.http.util.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.common.Strings;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.plugin.KNNPlugin;
import org.opensearch.rest.RestStatus;

import java.io.IOException;
import java.util.Map;

import static org.opensearch.knn.TestUtils.*;
import static org.opensearch.knn.common.KNNConstants.*;

public class TrainModelTestsIT extends AbstractRestartUpgradeTestCase {
    private static final String TRAINING_INDEX = KNN_BWC_PREFIX + "train-index";
    private static final String TRAINING_FIELD = "train-field";
    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSIONS = 5;
    private static final int K = 10;
    private static final int ADD_DOCS_CNT = 100;
    private static final String TEST_MODEL_ID = "test-model-id";
    private static final String MODEL_DESCRIPTION = "Description for train model test";

    // KNN train model test
    public void testKNNTrainModel() throws Exception {
        if (isRunningAgainstOldCluster()) {

            // Create a training index and randomly ingest data into it
            createBasicKnnIndex(TRAINING_INDEX, TRAINING_FIELD, DIMENSIONS);
            bulkIngestRandomVectors(TRAINING_INDEX, TRAINING_FIELD, ADD_DOCS_CNT, DIMENSIONS);
        } else {
            trainKNNModel(TEST_MODEL_ID, TRAINING_INDEX, TRAINING_FIELD, DIMENSIONS, MODEL_DESCRIPTION);
            verifyModel(TEST_MODEL_ID);

            createKnnIndex(testIndex, modelIndexMapping(TEST_FIELD, TEST_MODEL_ID));
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, ADD_DOCS_CNT);
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, 100, K);

            deleteKNNModel(TEST_MODEL_ID);
            deleteKNNIndex(testIndex);
            deleteKNNIndex(TRAINING_INDEX);
        }
    }

    // delete kNN model
    public void deleteKNNModel(String modelId) throws IOException {
        String restURI = String.join("/", KNNPlugin.KNN_BASE_URI, MODELS, modelId);
        Request request = new Request("DELETE", restURI);

        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));
    }

    // Confirm that the model gets created
    public void verifyModel(String modelId) throws IOException, InterruptedException {
        Response getResponse = getModel(modelId, null);
        String responseBody = EntityUtils.toString(getResponse.getEntity());
        assertNotNull(responseBody);

        Map<String, Object> responseMap = createParser(XContentType.JSON.xContent(), responseBody).map();
        assertEquals(modelId, responseMap.get(MODEL_ID));
        assertTrainingSucceeds(modelId, 30, 1000);
    }

    // train kNN model
    // method : "ivf", engine : "faiss", space_type : "l2"
    public void trainKNNModel(String modelId, String trainingIndexName, String trainingFieldName, int dimension, String description)
        throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, METHOD_IVF)
            .field(KNN_ENGINE, FAISS_NAME)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_NLIST, 1)
            .endObject()
            .endObject();
        Map<String, Object> method = xContentBuilderToMap(builder);

        Response trainResponse = trainModel(modelId, trainingIndexName, trainingFieldName, dimension, method, MODEL_DESCRIPTION);
        assertEquals(RestStatus.OK, RestStatus.fromCode(trainResponse.getStatusLine().getStatusCode()));
    }

    // mapping to create index from model
    public String modelIndexMapping(String fieldName, String modelId) throws IOException {
        return Strings.toString(
            XContentFactory.jsonBuilder()
                .startObject()
                .startObject(PROPERTIES)
                .startObject(fieldName)
                .field(VECTOR_TYPE, KNN_VECTOR)
                .field(MODEL_ID, modelId)
                .endObject()
                .endObject()
                .endObject()
        );
    }

    // train kNN model
    // method : "ivf", engine : "faiss", space_type : "l2", encoder : "pq" { code_size : 2, m : 2 }
    // public void trainKNNModel(String modelId, String trainingIndexName, String trainingFieldName, Integer dimension, String description)
    // throws IOException {
    //// XContentBuilder builder = XContentFactory.jsonBuilder()
    //// .startObject()
    //// .field(NAME, METHOD_IVF)
    //// .field(KNN_ENGINE, FAISS_NAME)
    //// .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
    //// .startObject(PARAMETERS)
    //// .field(METHOD_PARAMETER_NLIST, 1)
    //// .startObject(METHOD_ENCODER_PARAMETER)
    //// .field(NAME, ENCODER_PQ)
    //// .startObject(PARAMETERS)
    //// .field(ENCODER_PARAMETER_PQ_CODE_SIZE, 2)
    //// .field(ENCODER_PARAMETER_PQ_M, 2)
    //// .endObject()
    //// .endObject()
    //// .endObject()
    //// .endObject();
    // XContentBuilder builder = XContentFactory.jsonBuilder()
    // .startObject()
    // .field(NAME, "hnsw")
    // .field(KNN_ENGINE, FAISS_NAME)
    // .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
    // .startObject(PARAMETERS)
    // //.field(METHOD_PARAMETER_NLIST, 1)
    // .startObject(METHOD_ENCODER_PARAMETER)
    // .field(NAME, ENCODER_PQ)
    // .startObject(PARAMETERS)
    // .field(ENCODER_PARAMETER_PQ_CODE_SIZE, 8)
    // .field(ENCODER_PARAMETER_PQ_M, 8)
    // .endObject()
    // .endObject()
    // .endObject()
    // .endObject();
    // Map<String, Object> method = xContentBuilderToMap(builder);
    //
    // Response trainResponse = trainModel(modelId, trainingIndexName, trainingFieldName, dimension, method, description);
    // assertEquals(RestStatus.OK, RestStatus.fromCode(trainResponse.getStatusLine().getStatusCode()));
    // }

}
