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

package org.opensearch.knn.index.codec.KNN940Codec;

import lombok.Builder;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.CompoundFormat;
import org.apache.lucene.codecs.DocValuesFormat;
import org.apache.lucene.codecs.FilterCodec;
import org.opensearch.knn.index.codec.KNNFormatFacade;
import org.opensearch.knn.index.codec.KNNFormatFactory;

import static org.opensearch.knn.index.codec.KNNCodecFactory.CodecDelegateFactory.createKNN94DefaultDelegate;

public class KNN940Codec extends FilterCodec {
    private static final String KNN940 = "KNN940Codec";
    private final KNNFormatFacade knnFormatFacade;

    public KNN940Codec() {
        this(createKNN94DefaultDelegate());
    }

    /**
     * Sole constructor. When subclassing this codec, create a no-arg ctor and pass the delegate codec
     * and a unique name to this ctor.
     *
     * @param delegate
     */
    @Builder
    protected KNN940Codec(Codec delegate) {
        super(KNN940, delegate);
        knnFormatFacade = KNNFormatFactory.createKNN940Format(delegate);
    }

    @Override
    public DocValuesFormat docValuesFormat() {
        return knnFormatFacade.docValuesFormat();
    }

    @Override
    public CompoundFormat compoundFormat() {
        return knnFormatFacade.compoundFormat();
    }
}
