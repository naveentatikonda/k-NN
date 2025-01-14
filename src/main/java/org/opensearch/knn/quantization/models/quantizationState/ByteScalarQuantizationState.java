/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.models.quantizationState;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import org.apache.lucene.util.RamUsageEstimator;
import org.opensearch.Version;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.knn.quantization.models.quantizationParams.QuantizationParams;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;

import java.io.IOException;

@Getter
@NoArgsConstructor
@AllArgsConstructor
public class ByteScalarQuantizationState implements QuantizationState {
    private ScalarQuantizationParams quantizationParams;
    private byte[] indexTemplate;

    @Override
    public QuantizationParams getQuantizationParams() {
        return quantizationParams;
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeVInt(Version.CURRENT.id); // Write the version
        quantizationParams.writeTo(out);
        out.writeByteArray(indexTemplate);
    }

    public ByteScalarQuantizationState(StreamInput in) throws IOException {
        int version = in.readVInt(); // Read the version
        this.quantizationParams = new ScalarQuantizationParams(in, version);
        this.indexTemplate = in.readByteArray();

    }

    @Override
    public byte[] toByteArray() throws IOException {
        return QuantizationStateSerializer.serialize(this);
    }

    public static ByteScalarQuantizationState fromByteArray(final byte[] bytes) throws IOException {
        return (ByteScalarQuantizationState) QuantizationStateSerializer.deserialize(bytes, ByteScalarQuantizationState::new);
    }

    @Override
    public int getBytesPerVector() {
        return 0;
    }

    @Override
    public int getDimensions() {
        return 0;
    }

    @Override
    public long ramBytesUsed() {
        long size = RamUsageEstimator.shallowSizeOfInstance(ByteScalarQuantizationState.class);
        size += RamUsageEstimator.shallowSizeOf(quantizationParams);
        size += RamUsageEstimator.sizeOf(indexTemplate);
        return size;
    }
}
