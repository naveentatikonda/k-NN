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
import java.util.Arrays;

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

    // Train using min-max technique
    // @Override
    // public QuantizationState train(TrainingRequest<float[]> trainingRequest) throws IOException {
    // int[] sampledIndices = sampler.sample(trainingRequest.getTotalNumberOfVectors(), samplingSize);
    // Pair<float[], float[]> minAndMax = QuantizerHelper.calculateMinAndMax(trainingRequest, sampledIndices);
    // float[] diff = calculateDiff(minAndMax.getA(), minAndMax.getB());
    // ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.EIGHT_BIT);
    // return new ByteScalarQuantizationState(params, minAndMax.getA(), diff);
    // }

    // Train using mean standard deviation
    // @Override
    // public QuantizationState train(TrainingRequest<float[]> trainingRequest) throws IOException {
    // int[] sampledIndices = sampler.sample(trainingRequest.getTotalNumberOfVectors(), samplingSize);
    // Pair<float[], float[]> meanAndStdDev = QuantizerHelper.calculateMeanAndStdDev(trainingRequest, sampledIndices);
    // Pair<float[], float[]> minAndDiff = calculateMinAndDiff(meanAndStdDev.getA(), meanAndStdDev.getB());
    // ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.EIGHT_BIT);
    // return new ByteScalarQuantizationState(params, minAndDiff.getA(), minAndDiff.getB());
    // }

    @Override
    public QuantizationState train(TrainingRequest<float[]> trainingRequest) throws IOException {
        int[] sampledIndices = sampler.sample(trainingRequest.getTotalNumberOfVectors(), samplingSize);
        float[][] transposedVec = transposeVectors(trainingRequest, sampledIndices);
        Pair<float[], float[]> minAndDiff = calculateMinAndDiffUsingQuantile(transposedVec);
        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.EIGHT_BIT);
        return new ByteScalarQuantizationState(params, minAndDiff.getA(), minAndDiff.getB());
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

    private Pair<float[], float[]> calculateMinAndDiff(final float[] meanArray, final float[] stdArray) {
        int dimension = meanArray.length;
        float rs_arg = 1.0f;
        float[] diffArray = new float[dimension];
        float[] minArray = new float[dimension];

        for (int i = 0; i < dimension; i++) {
            minArray[i] = meanArray[i] - stdArray[i] * rs_arg;
            float max = meanArray[i] + stdArray[i] * rs_arg;
            diffArray[i] = max - minArray[i];
        }
        return new Pair<>(minArray, diffArray);
    }

    private Pair<float[], float[]> calculateMinAndDiffUsingQuantile(float[][] transposedVectors) {
        int dimension = transposedVectors.length;
        float rs_arg = 2f;
        float[] diffArray = new float[dimension];
        float[] minArray = new float[dimension];
        int n = transposedVectors[0].length;

        for (int i = 0; i < dimension; i++) {
            float[] vec = transposedVectors[i];
            Arrays.sort(vec);
            int o = (int) (rs_arg * n);
            if (o < 0) o = 0;
            if (o > n - o) o = n / 2;
            minArray[i] = vec[o];
            diffArray[i] = vec[n - 1 - o] - minArray[i];
        }
        return new Pair<>(minArray, diffArray);
    }

    private float[][] transposeVectors(TrainingRequest<float[]> trainingRequest, int[] sampledIndices) throws IOException {
        int totalSamples = sampledIndices.length;
        if (totalSamples > 0) {
            float[] vector = trainingRequest.getVectorAtThePosition(sampledIndices[0]);
            if (vector == null) {
                throw new IllegalArgumentException("Vector at sampled index " + sampledIndices[0] + " is null.");
            }
        }
        int dimension = trainingRequest.getVectorAtThePosition(sampledIndices[0]).length;
        float[][] transposedVec = new float[dimension][totalSamples];

        for (int i = 0; i < totalSamples; i++) {
            float[] vector = trainingRequest.getVectorAtThePosition(sampledIndices[0]);
            for (int j = 0; j < dimension; j++) {
                transposedVec[j][i] = vector[j];
            }
        }
        return transposedVec;
    }

    private void validateState(final QuantizationState state) {
        if (!(state instanceof ByteScalarQuantizationState)) {
            throw new IllegalArgumentException("Quantization state must be of type ByteScalarQuantizationState.");
        }
    }
}
