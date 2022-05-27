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

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.action.FailedNodeException;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.nodes.TransportNodesAction;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.inject.Inject;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.knn.indices.ModelCache;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.TransportService;

import java.io.IOException;
import java.util.List;

/**
 * Transport action to cancel model training on node in the cluster
 */
public class CancelModelTrainingTransportAction extends TransportNodesAction<
    CancelModelTrainingRequest,
    CancelModelTrainingResponse,
    CancelModelTrainingNodeRequest,
    CancelModelTrainingNodeResponse> {

    private static Logger logger = LogManager.getLogger(CancelModelTrainingTransportAction.class);

    @Inject
    public CancelModelTrainingTransportAction(
        ThreadPool threadPool,
        ClusterService clusterService,
        TransportService transportService,
        ActionFilters actionFilters
    ) {
        super(
            CancelModelTrainingAction.NAME,
            threadPool,
            clusterService,
            transportService,
            actionFilters,
            CancelModelTrainingRequest::new,
            CancelModelTrainingNodeRequest::new,
            ThreadPool.Names.SAME,
            CancelModelTrainingNodeResponse.class
        );
    }

    @Override
    protected CancelModelTrainingResponse newResponse(
        CancelModelTrainingRequest nodesRequest,
        List<CancelModelTrainingNodeResponse> responses,
        List<FailedNodeException> failures
    ) {
        return new CancelModelTrainingResponse(clusterService.getClusterName(), responses, failures);
    }

    @Override
    protected CancelModelTrainingNodeRequest newNodeRequest(CancelModelTrainingRequest request) {
        return new CancelModelTrainingNodeRequest(request.getModelId());
    }

    @Override
    protected CancelModelTrainingNodeResponse newNodeResponse(StreamInput in) throws IOException {
        return new CancelModelTrainingNodeResponse(in);
    }

    @Override
    protected CancelModelTrainingNodeResponse nodeOperation(CancelModelTrainingNodeRequest nodeRequest) {
        logger.debug("[KNN] Removing model \"" + nodeRequest.getModelId() + "\" on node \"" + clusterService.localNode().getId() + ".");
        ModelCache.getInstance().remove(nodeRequest.getModelId());
        return new CancelModelTrainingNodeResponse(clusterService.localNode());
    }
}
