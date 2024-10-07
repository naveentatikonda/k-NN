/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.iterators;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.util.BitSet;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.query.KNNQuery;
import org.opensearch.knn.index.query.SegmentLevelQuantizationInfo;
import org.opensearch.knn.index.vectorvalues.KNNBinaryVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNByteVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNFloatVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;

import java.io.IOException;

public class KNNIteratorFactory {
    public static KNNIterator getKNNIterator(
        BitSet matchedDocs,
        KNNQuery knnQuery,
        FieldInfo fieldInfo,
        SegmentReader reader,
        SpaceType spaceType,
        byte[] quantizedQueryVector,
        SegmentLevelQuantizationInfo segmentLevelQuantizationInfo,
        boolean isNestedRequired,
        LeafReaderContext leafReaderContext
    ) throws IOException {
        VectorDataType vectorDataType = FieldInfoExtractor.extractVectorDataType(fieldInfo);
        switch (vectorDataType) {
            case FLOAT:
                final KNNVectorValues<float[]> floatVectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader);
                if (isNestedRequired) {
                    return new NestedVectorIdsKNNIterator(
                        matchedDocs,
                        knnQuery.getQueryVector(),
                        (KNNFloatVectorValues) floatVectorValues,
                        spaceType,
                        knnQuery.getParentsFilter().getBitSet(leafReaderContext),
                        quantizedQueryVector,
                        segmentLevelQuantizationInfo
                    );
                }
                return new VectorIdsKNNIterator(
                    matchedDocs,
                    knnQuery.getQueryVector(),
                    (KNNFloatVectorValues) floatVectorValues,
                    spaceType,
                    quantizedQueryVector,
                    segmentLevelQuantizationInfo
                );
            case BYTE:
                final KNNVectorValues<byte[]> byteVectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader);
                if (isNestedRequired) {
                    return new NestedByteVectorIdsKNNIterator(
                        matchedDocs,
                        knnQuery.getQueryVector(),
                        (KNNByteVectorValues) byteVectorValues,
                        spaceType,
                        knnQuery.getParentsFilter().getBitSet(leafReaderContext)
                    );
                }
                return new ByteVectorIdsKNNIterator(
                    matchedDocs,
                    knnQuery.getQueryVector(),
                    (KNNByteVectorValues) byteVectorValues,
                    spaceType
                );
            case BINARY:
                final KNNVectorValues<byte[]> binaryVectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader);
                if (isNestedRequired) {
                    return new NestedBinaryVectorIdsKNNIterator(
                        matchedDocs,
                        knnQuery.getByteQueryVector(),
                        (KNNBinaryVectorValues) binaryVectorValues,
                        spaceType,
                        knnQuery.getParentsFilter().getBitSet(leafReaderContext)
                    );
                }
                return new BinaryVectorIdsKNNIterator(
                    matchedDocs,
                    knnQuery.getByteQueryVector(),
                    (KNNBinaryVectorValues) binaryVectorValues,
                    spaceType
                );
        }

        return null;
    }
}
