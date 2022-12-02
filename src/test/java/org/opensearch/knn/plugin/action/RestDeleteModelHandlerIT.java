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

package org.opensearch.knn.plugin.action;

import org.apache.http.util.EntityUtils;
import org.opensearch.action.DocWriteResponse;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.util.KNNEngine;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelState;
import org.opensearch.knn.plugin.KNNPlugin;
import org.opensearch.knn.plugin.transport.DeleteModelResponse;
import org.opensearch.rest.RestStatus;

import java.io.IOException;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.MODELS;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.common.KNNConstants.MODEL_INDEX_NAME;

/**
 * Integration tests to check the correctness of {@link org.opensearch.knn.plugin.rest.RestDeleteModelHandler}
 */

public class RestDeleteModelHandlerIT extends KNNRestTestCase {

    private ModelMetadata getModelMetadata() {
        return new ModelMetadata(KNNEngine.DEFAULT, SpaceType.DEFAULT, 4, ModelState.CREATED, "2021-03-27", "test model", "");
    }

    public void testDeleteModelExists() throws IOException {
        createModelSystemIndex();
        String testModelID = "test-model-id";
        byte[] testModelBlob = "hello".getBytes();
        ModelMetadata testModelMetadata = getModelMetadata();

        addModelToSystemIndex(testModelID, testModelMetadata, testModelBlob);
        assertEquals(getDocCount(MODEL_INDEX_NAME), 1);

        String restURI = String.join("/", KNNPlugin.KNN_BASE_URI, MODELS, testModelID);
        Request request = new Request("DELETE", restURI);

        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        assertEquals(getDocCount(MODEL_INDEX_NAME), 0);
    }

    public void testDeleteModelFailsInvalid() throws IOException {
        createModelSystemIndex();
        String restURI = String.join("/", KNNPlugin.KNN_BASE_URI, MODELS, "invalid-model-id");
        Request request = new Request("DELETE", restURI);

        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));
        String responseBody = EntityUtils.toString(response.getEntity());
        assertNotNull(responseBody);

        Map<String, Object> responseMap = createParser(XContentType.JSON.xContent(), responseBody).map();

        assertEquals("invalid-model-id", responseMap.get(MODEL_ID));
        assertEquals(DocWriteResponse.Result.NOT_FOUND.getLowercase(), responseMap.get(DeleteModelResponse.RESULT));
        assertNotNull(responseMap.get(DeleteModelResponse.ERROR_MSG));
    }
}
