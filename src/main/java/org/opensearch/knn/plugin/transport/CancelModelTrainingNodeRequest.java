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

package org.opensearch.knn.plugin.transport;

import org.opensearch.action.support.nodes.BaseNodeRequest;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;

import java.io.IOException;

/**
 * Request sent to each node to cancel model training if that model is being trained on that node
 */
public class CancelModelTrainingNodeRequest extends BaseNodeRequest {

    private final String modelId;

    /**
     * Constructor.
     *
     * @param modelId identifier of the model.
     */
    public CancelModelTrainingNodeRequest(String modelId) {
        super();
        this.modelId = modelId;
    }

    /**
     * Constructor from stream
     *
     * @param in input stream
     * @throws IOException thrown when reading from stream fails
     */
    public CancelModelTrainingNodeRequest(StreamInput in) throws IOException {
        super(in);
        this.modelId = in.readString();
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
        out.writeString(modelId);
    }

    /**
     * Getter for model id
     *
     * @return modelId
     */
    public String getModelId() {
        return modelId;
    }
}
