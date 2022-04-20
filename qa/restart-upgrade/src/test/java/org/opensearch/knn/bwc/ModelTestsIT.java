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
import org.opensearch.client.Response;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.indices.Model;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelState;

import java.io.IOException;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.*;

public class ModelTestsIT extends AbstractRestartUpgradeTestCase {
    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSIONS = 2;
    private static final int K = 5;
    private static final int ADD_DOCS_CNT = 10;

    // KNN
    public void testKNNModel() throws Exception {
        if (isRunningAgainstOldCluster()) {
            String modelId = "test_model";
            byte[] modelBlob = "hello".getBytes();

            // createIndex(MODEL_INDEX_NAME);
            Model model = new Model(
                new ModelMetadata(
                    KNNEngine.DEFAULT,
                    SpaceType.DEFAULT,
                    DIMENSIONS,
                    ModelState.CREATED,
                    ZonedDateTime.now(ZoneOffset.UTC).toString(),
                    "",
                    ""
                ),
                modelBlob,
                modelId
            );
            // verifyModel(modelId);
            // addDoc(model);

        } else {

            // deleteKNNIndex(testIndex);
        }
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

    // public void addDoc(Model model) throws IOException, ExecutionException, InterruptedException {
    // ModelMetadata modelMetadata = model.getModelMetadata();
    //
    // XContentBuilder builder = XContentFactory.jsonBuilder()
    // .startObject()
    // .field(MODEL_ID, model.getModelID())
    // .field(KNN_ENGINE, modelMetadata.getKnnEngine().getName())
    // .field(METHOD_PARAMETER_SPACE_TYPE, modelMetadata.getSpaceType().getValue())
    // .field(DIMENSION, modelMetadata.getDimension())
    // .field(MODEL_STATE, modelMetadata.getState().getName())
    // .field(MODEL_TIMESTAMP, modelMetadata.getTimestamp().toString())
    // .field(MODEL_DESCRIPTION, modelMetadata.getDescription())
    // .field(MODEL_ERROR, modelMetadata.getError());
    //
    // if (model.getModelBlob() != null) {
    // builder.field(MODEL_BLOB_PARAMETER, Base64.getEncoder().encodeToString(model.getModelBlob()));
    // }
    //
    // builder.endObject();
    //
    // IndexRequest indexRequest = new IndexRequest().index(MODEL_INDEX_NAME)
    // .id(model.getModelID())
    // .source(builder)
    // .setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);
    //
    // IndexResponse response = client().index(indexRequest).get();
    // assertTrue(response.status() == RestStatus.CREATED || response.status() == RestStatus.OK);
    // }

}
