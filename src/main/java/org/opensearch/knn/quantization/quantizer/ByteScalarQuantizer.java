/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.quantizer;

import org.apache.lucene.index.FieldInfo;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.core.common.bytes.BytesArray;
import org.opensearch.core.xcontent.DeprecationHandler;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.transfer.OffHeapVectorTransfer;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.jni.JNIService;
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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.opensearch.knn.common.FieldInfoExtractor.extractVectorDataType;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.index.codec.transfer.OffHeapVectorTransferFactory.getVectorTransfer;

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

    // Train using Quantile
    // @Override
    // public QuantizationState train(TrainingRequest<float[]> trainingRequest) throws IOException {
    // int[] sampledIndices = sampler.sample(trainingRequest.getTotalNumberOfVectors(), samplingSize);
    // float[][] transposedVec = transposeVectors(trainingRequest, sampledIndices);
    // Pair<float[], float[]> minAndDiff = calculateMinAndDiffUsingQuantile(transposedVec);
    // ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.EIGHT_BIT);
    // return new ByteScalarQuantizationState(params, minAndDiff.getA(), minAndDiff.getB());
    // }

    @Override
    public QuantizationState train(final TrainingRequest<float[]> trainingRequest) throws IOException {
        return null;
    }

    @Override
    public QuantizationState train(final TrainingRequest<float[]> trainingRequest, final FieldInfo fieldInfo) throws IOException {
        int[] sampledIndices = sampler.sample(trainingRequest.getTotalNumberOfVectors(), samplingSize);
        if (sampledIndices.length == 0) {
            return null;
        }
        float[] vector = trainingRequest.getVectorAtThePosition(sampledIndices[0]);
        if (vector == null) {
            throw new IllegalArgumentException("Vector at sampled index " + sampledIndices[0] + " is null.");
        }
        int dimension = vector.length;
        byte[] indexTemplate = JNIService.trainIndex(
            getParameters(fieldInfo),
            dimension,
            getVectorAddressOfTrainData(sampledIndices, fieldInfo, trainingRequest, dimension),
            KNNEngine.FAISS
        );

        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.EIGHT_BIT);
        return new ByteScalarQuantizationState(params, indexTemplate);
    }

    private long getVectorAddressOfTrainData(
        int[] sampledIndices,
        FieldInfo fieldInfo,
        final TrainingRequest<float[]> trainingRequest,
        int dimension
    ) throws IOException {
        int totalSamples = sampledIndices.length;

        final OffHeapVectorTransfer vectorTransfer = getVectorTransfer(
            extractVectorDataType(fieldInfo),
            4 * dimension,
            sampledIndices.length
        );
        final List<Integer> transferredDocIds = new ArrayList<>(totalSamples);
        for (int i = 0; i < totalSamples; i++) {
            Object vectorToTransfer = trainingRequest.getVectorAtThePosition(sampledIndices[i]);
            vectorTransfer.transfer(vectorToTransfer, true);
        }
        vectorTransfer.flush(true);
        return vectorTransfer.getVectorAddress();
    }

    private Map<String, Object> getParameters(final FieldInfo fieldInfo) throws IOException {
        Map<String, Object> parameters = new HashMap<>();
        Map<String, String> fieldAttributes = fieldInfo.attributes();
        String parametersString = fieldAttributes.get(KNNConstants.PARAMETERS);

        // parametersString will be null when legacy mapper is used
        if (parametersString == null) {
            parameters.put(KNNConstants.SPACE_TYPE, fieldAttributes.getOrDefault(KNNConstants.SPACE_TYPE, SpaceType.DEFAULT.getValue()));

            Map<String, Object> algoParams = new HashMap<>();

            String efConstruction = fieldAttributes.get(KNNConstants.HNSW_ALGO_EF_CONSTRUCTION);
            if (efConstruction != null) {
                algoParams.put(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION, Integer.parseInt(efConstruction));
            }

            String m = fieldAttributes.get(KNNConstants.HNSW_ALGO_M);
            if (m != null) {
                algoParams.put(KNNConstants.METHOD_PARAMETER_M, Integer.parseInt(m));
            }
            parameters.put(PARAMETERS, algoParams);
        } else {
            parameters.putAll(
                XContentHelper.createParser(
                    NamedXContentRegistry.EMPTY,
                    DeprecationHandler.THROW_UNSUPPORTED_OPERATION,
                    new BytesArray(parametersString),
                    MediaTypeRegistry.getDefaultMediaType()
                ).map()
            );
        }

        parameters.put(KNNConstants.VECTOR_DATA_TYPE_FIELD, VectorDataType.FLOAT);
        // parameters.put("rangestat", 1);
        // parameters.put("rs", 1);
        // parameters.put("rangestat_arg", 20);
        // parameters.put("rs_arg", 10.5);

        // Used to determine how many threads to use when indexing
        parameters.put(KNNConstants.INDEX_THREAD_QTY, KNNSettings.state().getSettingValue(KNNSettings.KNN_ALGO_PARAM_INDEX_THREAD_QTY));

        return parameters;
    }

    @Override
    public void quantize(float[] vector, QuantizationState state, QuantizationOutput<byte[]> output) {
        // if (vector == null) {
        // throw new IllegalArgumentException("Vector to quantize must not be null.");
        // }
        // validateState(state);
        // int vectorLength = vector.length;
        // ByteScalarQuantizationState byteSQState = (ByteScalarQuantizationState) state;
        // float[] minArray = byteSQState.getMin();
        // float[] diffArray = byteSQState.getDiff();
        // if (minArray == null || minArray.length != vectorLength || diffArray == null || diffArray.length != vectorLength) {
        // throw new IllegalArgumentException("min and diff arrays must not be null and must match the dimension of the vector.");
        // }
        // output.prepareQuantizedVector(vectorLength);
        // quantizeVector(vector, minArray, diffArray, output.getQuantizedVector());
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
