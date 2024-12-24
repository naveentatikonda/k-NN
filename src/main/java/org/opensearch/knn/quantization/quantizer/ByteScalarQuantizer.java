/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.quantizer;

import org.opensearch.knn.quantization.enums.ScalarQuantizationType;
import org.opensearch.knn.quantization.models.quantizationOutput.QuantizationOutput;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;
import org.opensearch.knn.quantization.models.quantizationState.ByteScalarQuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;
import org.opensearch.knn.quantization.models.requests.TrainingRequest;
import org.opensearch.knn.quantization.sampler.Sampler;
import org.opensearch.knn.quantization.sampler.SamplerType;
import org.opensearch.knn.quantization.sampler.SamplingFactory;
import oshi.util.tuples.Pair;

import java.io.IOException;

public class ByteScalarQuantizer implements Quantizer<float[], byte[]> {
    private final int bitsPerCoordinate;
    private final int samplingSize; // Sampling size for training
    private final Sampler sampler; // Sampler for training
    // Currently Lucene has sampling size as
    // 25000 for segment level training , Keeping same
    // to having consistent, Will revisit
    // if this requires change
    private static final int DEFAULT_SAMPLE_SIZE = 25000;

    public ByteScalarQuantizer(final int bitsPerCoordinate) {
        if (bitsPerCoordinate != 8) {
            throw new IllegalArgumentException("bitsPerCoordinate must be 8 for byte scalar quantizer.");
        }
        this.bitsPerCoordinate = bitsPerCoordinate;
        this.samplingSize = DEFAULT_SAMPLE_SIZE;
        this.sampler = SamplingFactory.getSampler(SamplerType.RESERVOIR);
    }

    @Override
    public QuantizationState train(TrainingRequest<float[]> trainingRequest) throws IOException {
        int[] sampledIndices = sampler.sample(trainingRequest.getTotalNumberOfVectors(), samplingSize);
        Pair<float[], float[]> minAndMax = QuantizerHelper.calculateMinAndMax(trainingRequest, sampledIndices);
        float[] diff = calculateDiff(minAndMax.getA(), minAndMax.getB());
        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.EIGHT_BIT);
        return new ByteScalarQuantizationState(params, minAndMax.getA(), diff);
    }

    @Override
    public void quantize(float[] vector, QuantizationState state, QuantizationOutput<byte[]> output) {
        if (vector == null) {
            throw new IllegalArgumentException("Vector to quantize must not be null.");
        }
        validateState(state);
        int vectorLength = vector.length;
        ByteScalarQuantizationState byteSQState = (ByteScalarQuantizationState) state;
        float[] minArray = byteSQState.getMin();
        float[] diffArray = byteSQState.getDiff();
        if (minArray == null || minArray.length != vectorLength || diffArray == null || diffArray.length != vectorLength) {
            throw new IllegalArgumentException("min and diff arrays must not be null and must match the dimension of the vector.");
        }
        output.prepareQuantizedVector(vectorLength);
        quantizeVector(vector, minArray, diffArray, output.getQuantizedVector());
    }

    private void quantizeVector(final float[] vector, final float[] min, final float[] diff, byte[] quantizedVector) {
        for (int i = 0; i < vector.length; i++) {
            float x = 0;
            if (diff[i] != 0) {
                x = (vector[i] - min[i]) / diff[i];
                x = Math.max(x, 0);
                x = Math.min(1, x);
            }
            quantizedVector[i] = (byte) ((x * 255) - 128);
        }
    }

    private float[] calculateDiff(final float[] minArray, final float[] maxArray) {
        int dimension = minArray.length;
        float rs_arg = 0;
        float[] diffArray = new float[dimension];

        for (int i = 0; i < dimension; i++) {
            float exp = (maxArray[i] - minArray[i]) * rs_arg;
            minArray[i] = minArray[i] - exp;
            maxArray[i] = maxArray[i] + exp;
            diffArray[i] = maxArray[i] - minArray[i];
        }
        return diffArray;
    }

    private void validateState(final QuantizationState state) {
        if (!(state instanceof ByteScalarQuantizationState)) {
            throw new IllegalArgumentException("Quantization state must be of type ByteScalarQuantizationState.");
        }
    }
}
