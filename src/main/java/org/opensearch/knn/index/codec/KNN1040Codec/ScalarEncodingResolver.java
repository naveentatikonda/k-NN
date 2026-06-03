/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.experimental.UtilityClass;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat.ScalarEncoding;

import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

/**
 * Single decision point that maps a document bit width (the number of bits used to quantize each
 * dimension of a stored vector) to the Lucene {@link ScalarEncoding} used by the FAISS SQ
 * memory-optimized-search path, and back.
 *
 * <p>The FAISS SQ path always uses a 4-bit (nibble) query, so the relevant encodings are exactly
 * those whose query bit width is {@link #QUERY_BITS}. In Lucene 10.4 these are:
 * <ul>
 *   <li>1-bit document → {@code SINGLE_BIT_QUERY_NIBBLE} (x32)</li>
 *   <li>2-bit document → {@code DIBIT_QUERY_NIBBLE} (x16)</li>
 *   <li>4-bit document → {@code PACKED_NIBBLE} (x8)</li>
 * </ul>
 *
 * <p>The mapping is resolved dynamically from {@link ScalarEncoding#getBits()} and
 * {@link ScalarEncoding#getQueryBits()} rather than by hardcoding enum names, so it stays correct
 * if Lucene renames constants and so adding a new supported bit width only requires updating
 * {@link #SUPPORTED_DOC_BITS}. This keeps the {@code SINGLE_BIT_QUERY_NIBBLE} reference out of the
 * codec format, the build strategy, and the scorer (Requirement 11.4).
 */
@UtilityClass
public class ScalarEncodingResolver {

    /** The FAISS SQ path always quantizes the query to a 4-bit nibble. */
    public static final int QUERY_BITS = 4;

    /** Document bit widths supported by the FAISS SQ memory-optimized-search path. */
    private static final int[] SUPPORTED_DOC_BITS = { 1, 2, 4 };

    /** docBits -> ScalarEncoding, resolved once from the enum metadata. */
    private static final Map<Integer, ScalarEncoding> DOC_BITS_TO_ENCODING = buildDocBitsToEncoding();

    private static Map<Integer, ScalarEncoding> buildDocBitsToEncoding() {
        final Map<Integer, ScalarEncoding> map = new HashMap<>();
        for (int docBits : SUPPORTED_DOC_BITS) {
            for (ScalarEncoding encoding : ScalarEncoding.values()) {
                if (encoding.getBits() == docBits && encoding.getQueryBits() == QUERY_BITS) {
                    map.put(docBits, encoding);
                    break;
                }
            }
            if (map.containsKey(docBits) == false) {
                throw new IllegalStateException(
                    String.format(
                        Locale.ROOT,
                        "No Lucene ScalarEncoding found for document bits=%d with a %d-bit query. "
                            + "The installed Lucene version may not expose this encoding.",
                        docBits,
                        QUERY_BITS
                    )
                );
            }
        }
        return map;
    }

    /**
     * Returns the {@link ScalarEncoding} used to store documents quantized to {@code docBits} bits
     * per dimension with a 4-bit nibble query.
     *
     * @param docBits the document bit width (1, 2, or 4)
     * @return the corresponding Lucene scalar encoding
     * @throws IllegalArgumentException if {@code docBits} is not a supported document bit width
     */
    public static ScalarEncoding forDocBits(final int docBits) {
        final ScalarEncoding encoding = DOC_BITS_TO_ENCODING.get(docBits);
        if (encoding == null) {
            throw new IllegalArgumentException(
                String.format(Locale.ROOT, "Unsupported SQ document bit width: %d. Supported: %s", docBits, supportedDocBitsString())
            );
        }
        return encoding;
    }

    /**
     * @param docBits the candidate document bit width
     * @return {@code true} if {@code docBits} maps to a supported FAISS SQ scalar encoding
     */
    public static boolean isSupportedDocBits(final int docBits) {
        return DOC_BITS_TO_ENCODING.containsKey(docBits);
    }

    /**
     * Inverse of {@link #forDocBits(int)} — the document bit width carried by {@code encoding}.
     *
     * @param encoding a Lucene scalar encoding
     * @return the document bit width ({@link ScalarEncoding#getBits()})
     */
    public static int docBits(final ScalarEncoding encoding) {
        return encoding.getBits();
    }

    /**
     * @param encoding a Lucene scalar encoding
     * @return {@code true} if {@code encoding} is one of the FAISS-SQ-supported nibble-query encodings
     */
    public static boolean isSupportedEncoding(final ScalarEncoding encoding) {
        return encoding != null && DOC_BITS_TO_ENCODING.containsValue(encoding);
    }

    private static String supportedDocBitsString() {
        final StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < SUPPORTED_DOC_BITS.length; i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(SUPPORTED_DOC_BITS[i]);
        }
        return sb.append(']').toString();
    }
}
