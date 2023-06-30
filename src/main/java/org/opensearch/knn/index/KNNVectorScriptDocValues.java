/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.util.BytesRef;
import org.opensearch.ExceptionsHelper;
import org.opensearch.index.fielddata.ScriptDocValues;
import org.opensearch.knn.index.codec.util.KNNVectorSerializer;
import org.opensearch.knn.index.codec.util.KNNVectorSerializerFactory;

import java.io.ByteArrayInputStream;
import java.io.IOException;

public final class KNNVectorScriptDocValues extends ScriptDocValues<float[]> {

    private final BinaryDocValues binaryDocValues;
    private final String fieldName;
    private boolean docExists;
    private final String vectorDataType;

    public KNNVectorScriptDocValues(BinaryDocValues binaryDocValues, String fieldName, String vectorDataType) {
        this.binaryDocValues = binaryDocValues;
        this.fieldName = fieldName;
        this.vectorDataType = vectorDataType;
    }

    @Override
    public void setNextDocId(int docId) throws IOException {
        if (binaryDocValues.advanceExact(docId)) {
            docExists = true;
            return;
        }
        docExists = false;
    }

    // public byte[] byteToByteIntArray(ByteArrayInputStream byteStream) throws IOException, ClassNotFoundException {
    // if (byteStream == null) {
    // throw new IllegalArgumentException("Byte stream cannot be deserialized to array of bytes");
    // }
    //// DataInputStream dataIs = new DataInputStream
    //// (byteStream);
    //// final byte[] vector = new byte[byteStream.available()];
    //// int i=0;
    //// // available stream to be read
    //// while(dataIs.available()>0)
    //// {
    ////
    //// vector[i++] = dataIs.readByte();
    ////
    //// }
    //
    // final ObjectInputStream objectStream = new ObjectInputStream(byteStream);
    // final byte[] vector = (byte[]) objectStream.readObject();
    //
    //// final byte[] vectorAsByteArray = new byte[byteStream.available()];
    //// byteStream.read(vectorAsByteArray, 0, byteStream.available());
    //// final byte[] vector = new byte[vectorAsByteArray.length];
    //// ByteBuffer.wrap(vectorAsByteArray).asFloatBuffer().get((byte)vector);
    //// byte a = vectorAsByteArray[0];
    // return vector;
    // }

    public float[] getValue() {
        if (!docExists) {
            String errorMessage = String.format(
                "One of the document doesn't have a value for field '%s'. "
                    + "This can be avoided by checking if a document has a value for the field or not "
                    + "by doc['%s'].size() == 0 ? 0 : {your script}",
                fieldName,
                fieldName
            );
            throw new IllegalStateException(errorMessage);
        }
        try {
            BytesRef value = binaryDocValues.binaryValue();
            if (vectorDataType.equals("byte")) {
                byte[] byteVector = value.bytes;
                float[] vector = new float[value.length];
                int i = 0;
                int j = value.offset;

                while (i < value.length) {
                    vector[i++] = value.bytes[j++];
                }
                return vector;
            } else {
                ByteArrayInputStream byteStream = new ByteArrayInputStream(value.bytes, value.offset, value.length);
                final KNNVectorSerializer vectorSerializer = KNNVectorSerializerFactory.getSerializerByStreamContent(byteStream);
                final float[] vector = vectorSerializer.byteToFloatArray(byteStream);
                return vector;
            }

        } catch (IOException e) {
            throw ExceptionsHelper.convertToOpenSearchException(e);
        }
    }

    @Override
    public int size() {
        return docExists ? 1 : 0;
    }

    @Override
    public float[] get(int i) {
        throw new UnsupportedOperationException("knn vector does not support this operation");
    }
}
