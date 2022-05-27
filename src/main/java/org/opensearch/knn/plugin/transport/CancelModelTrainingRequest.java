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

import org.opensearch.action.support.nodes.BaseNodesRequest;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;

import java.io.IOException;

/**
 * Request used to ask some or all nodes in the cluster to cancel model training.
 */
public class CancelModelTrainingRequest extends BaseNodesRequest<CancelModelTrainingRequest> {

    private final String modelId;

    /**
     * Constructor.
     *
     * @param modelId Id of model for which the training needs to be cancelled
     * @param nodeIds Id's of nodes
     */
    public CancelModelTrainingRequest(String modelId, String... nodeIds) {
        super(nodeIds);
        this.modelId = modelId;
    }

    /**
     * Constructor.
     *
     * @param in input stream
     * @throws IOException thrown when reading input stream fails
     */
    public CancelModelTrainingRequest(StreamInput in) throws IOException {
        super(in);
        this.modelId = in.readString();
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
        out.writeString(modelId);
    }

    /**
     * Get model id request
     *
     * @return modelId
     */
    public String getModelId() {
        return modelId;
    }
}
