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

package org.opensearch.knn.index.codec;

@FunctionalInterface
public interface Function5Arity<S, T, U, V, X, R> {
    R apply(S s, T t, U u, V v, X x);
}
