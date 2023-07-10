/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import lombok.SneakyThrows;
import org.apache.lucene.document.BinaryDocValuesField;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.analysis.MockAnalyzer;
import org.junit.Assert;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil;

import java.io.IOException;
import java.util.Locale;

import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;

public class VectorDataTypeTests extends KNNTestCase {

    private static final String MOCK_FLOAT_INDEX_FIELD_NAME = "test-float-index-field-name";
    private static final String MOCK_BYTE_INDEX_FIELD_NAME = "test-byte-index-field-name";
    private static final float[] SAMPLE_FLOAT_VECTOR_DATA = new float[] { 10.0f, 25.0f };
    private static final byte[] SAMPLE_BYTE_VECTOR_DATA = new byte[] { 10, 25 };
    private Directory directory;
    private DirectoryReader reader;

    @SneakyThrows
    public void testGetDocValuesWithFloatVectorDataType() {
        KNNVectorScriptDocValues scriptDocValues = getKNNFloatVectorScriptDocValues();

        scriptDocValues.setNextDocId(0);
        Assert.assertArrayEquals(SAMPLE_FLOAT_VECTOR_DATA, scriptDocValues.getValue(), 0.1f);

        reader.close();
        directory.close();
    }

    @SneakyThrows
    public void testGetDocValuesWithByteVectorDataType() {
        KNNVectorScriptDocValues scriptDocValues = getKNNByteVectorScriptDocValues();

        scriptDocValues.setNextDocId(0);
        Assert.assertArrayEquals(SAMPLE_FLOAT_VECTOR_DATA, scriptDocValues.getValue(), 0.1f);

        reader.close();
        directory.close();
    }

    public void testFloatVectorValueValidations() {
        // Validate Float Vector Value which is NaN and throws exception
        IllegalArgumentException ex = expectThrows(
            IllegalArgumentException.class,
            () -> KNNVectorFieldMapperUtil.validateFloatVectorValue(Float.NaN)
        );
        assertTrue(ex.getMessage().contains("KNN vector values cannot be NaN"));

        // Validate Float Vector Value which is infinite and throws exception
        IllegalArgumentException ex1 = expectThrows(
            IllegalArgumentException.class,
            () -> KNNVectorFieldMapperUtil.validateFloatVectorValue(Float.POSITIVE_INFINITY)
        );
        assertTrue(ex1.getMessage().contains("KNN vector values cannot be infinity"));
    }

    public void testByteVectorValueValidations() {
        // Validate Byte Vector Value which is float with decimal values and throws exception
        IllegalArgumentException ex = expectThrows(
            IllegalArgumentException.class,
            () -> KNNVectorFieldMapperUtil.validateByteVectorValue(10.54f)
        );
        assertTrue(
            ex.getMessage()
                .contains(
                    String.format(
                        Locale.ROOT,
                        "[%s] field was set as [%s] in index mapping. But, KNN vector values are floats instead of byte integers",
                        VECTOR_DATA_TYPE_FIELD,
                        VectorDataType.BYTE.getValue()
                    )
                )
        );

        // Validate Byte Vector Value which is not in the byte range and throws exception
        IllegalArgumentException ex1 = expectThrows(
            IllegalArgumentException.class,
            () -> KNNVectorFieldMapperUtil.validateByteVectorValue(200f)
        );
        assertTrue(
            ex1.getMessage()
                .contains(
                    String.format(
                        Locale.ROOT,
                        "[%s] field was set as [%s] in index mapping. But, KNN vector values are not within in the byte range [%d, %d]",
                        VECTOR_DATA_TYPE_FIELD,
                        VectorDataType.BYTE.getValue(),
                        Byte.MIN_VALUE,
                        Byte.MAX_VALUE
                    )
                )
        );
    }

    @SneakyThrows
    private KNNVectorScriptDocValues getKNNFloatVectorScriptDocValues() {
        directory = newDirectory();
        createKNNFloatVectorDocument(directory);
        reader = DirectoryReader.open(directory);
        LeafReaderContext leafReaderContext = reader.getContext().leaves().get(0);
        return new KNNVectorScriptDocValues(
            leafReaderContext.reader().getBinaryDocValues(VectorDataTypeTests.MOCK_FLOAT_INDEX_FIELD_NAME),
            VectorDataTypeTests.MOCK_FLOAT_INDEX_FIELD_NAME,
            VectorDataType.FLOAT
        );
    }

    @SneakyThrows
    private KNNVectorScriptDocValues getKNNByteVectorScriptDocValues() {
        directory = newDirectory();
        createKNNByteVectorDocument(directory);
        reader = DirectoryReader.open(directory);
        LeafReaderContext leafReaderContext = reader.getContext().leaves().get(0);
        return new KNNVectorScriptDocValues(
            leafReaderContext.reader().getBinaryDocValues(VectorDataTypeTests.MOCK_BYTE_INDEX_FIELD_NAME),
            VectorDataTypeTests.MOCK_BYTE_INDEX_FIELD_NAME,
            VectorDataType.BYTE
        );
    }

    private void createKNNFloatVectorDocument(Directory directory) throws IOException {
        IndexWriterConfig conf = newIndexWriterConfig(new MockAnalyzer(random()));
        IndexWriter writer = new IndexWriter(directory, conf);
        Document knnDocument = new Document();
        knnDocument.add(
            new BinaryDocValuesField(
                MOCK_FLOAT_INDEX_FIELD_NAME,
                new VectorField(MOCK_FLOAT_INDEX_FIELD_NAME, SAMPLE_FLOAT_VECTOR_DATA, new FieldType()).binaryValue()
            )
        );
        writer.addDocument(knnDocument);
        writer.commit();
        writer.close();
    }

    private void createKNNByteVectorDocument(Directory directory) throws IOException {
        IndexWriterConfig conf = newIndexWriterConfig(new MockAnalyzer(random()));
        IndexWriter writer = new IndexWriter(directory, conf);
        Document knnDocument = new Document();
        knnDocument.add(
            new BinaryDocValuesField(
                MOCK_BYTE_INDEX_FIELD_NAME,
                new VectorField(MOCK_BYTE_INDEX_FIELD_NAME, SAMPLE_BYTE_VECTOR_DATA, new FieldType()).binaryValue()
            )
        );
        writer.addDocument(knnDocument);
        writer.commit();
        writer.close();
    }
}
