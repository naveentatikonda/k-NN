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

import org.opensearch.action.ActionType;
import org.opensearch.common.io.stream.Writeable;

/**
 * Action used to cancel model training on some node
 */
public class CancelModelTrainingAction extends ActionType<CancelModelTrainingResponse> {

    public static final String NAME = "cluster:admin/knn_cancel_model_training_action";
    public static final CancelModelTrainingAction INSTANCE = new CancelModelTrainingAction(NAME, CancelModelTrainingResponse::new);

    /**
     * Constructor
     *
     * @param name name of action
     * @param responseReader reader for the CancelModelTrainingResponse response
     */
    public CancelModelTrainingAction(String name, Writeable.Reader<CancelModelTrainingResponse> responseReader) {
        super(name, responseReader);
    }
}
