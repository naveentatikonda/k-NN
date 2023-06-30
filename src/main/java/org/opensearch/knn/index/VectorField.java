/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.apache.lucene.document.Field;
import org.apache.lucene.index.IndexableFieldType;
import org.apache.lucene.util.BytesRef;
import org.opensearch.knn.index.codec.util.KNNVectorSerializer;
import org.opensearch.knn.index.codec.util.KNNVectorSerializerFactory;

public class VectorField extends Field {

    public VectorField(String name, float[] value, IndexableFieldType type) {
        super(name, new BytesRef(), type);
        try {
            final KNNVectorSerializer vectorSerializer = KNNVectorSerializerFactory.getDefaultSerializer();
            final byte[] floatToByte = vectorSerializer.floatToByteArray(value);
            this.setBytesValue(floatToByte);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public VectorField(String name, byte[] value, IndexableFieldType type) {
        super(name, new BytesRef(), type);
        try {
            // final KNNVectorSerializer vectorSerializer = KNNVectorSerializerFactory.getDefaultSerializer();
            // byte[] bytes = new byte[value.length];
            //
            // ByteBuffer byteBuffer = ByteBuffer.wrap(bytes);
            // for (int i=0; i<value.length; i++) {
            // byteBuffer.put(value[i]);
            // }
            // this.setBytesValue(bytes);

            // final byte[] floatToByte = vectorSerializer.floatToByteArray(value);

            // public byte[] floatToByteArray(byte[] input) {
            // final ByteBuffer bb = ByteBuffer.allocate(value.length).order(ByteOrder.BIG_ENDIAN);
            // IntStream.range(0, value.length).forEach((index) -> bb.put(value[index]));
            // byte[] bytes = new byte[bb.flip().limit()];
            // bb.get(bytes);
            // this.setBytesValue(bytes);
            // return bytes;
            // }
            this.setBytesValue(value);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
