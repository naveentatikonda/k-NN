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

/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class org_opensearch_knn_jni_FaissService */

#ifndef _Included_org_opensearch_knn_jni_FaissService
#define _Included_org_opensearch_knn_jni_FaissService
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Class:     org_opensearch_knn_jni_FaissService
 * Method:    createIndex
 * Signature: ([IJILjava/lang/String;Ljava/util/Map;)V
 */
JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_createIndex
        (JNIEnv *, jclass, jintArray, jlong, jint, jstring, jobject);

/*
 * Class:     org_opensearch_knn_jni_FaissService
 * Method:    createBinaryIndex
 * Signature: ([IJILjava/lang/String;Ljava/util/Map;)V
 */
JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_createBinaryIndex
        (JNIEnv *, jclass, jintArray, jlong, jint, jstring, jobject);

/*
 * Class:     org_opensearch_knn_jni_FaissService
 * Method:    createByteIndex
 * Signature: ([IJILjava/lang/String;Ljava/util/Map;)V
 */
JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_createByteIndex
        (JNIEnv *, jclass, jintArray, jlong, jint, jstring, jobject);

/*
 * Class:     org_opensearch_knn_jni_FaissService
 * Method:    createIndexFromTemplate
 * Signature: ([IJILjava/lang/String;[BLjava/util/Map;)V
 */
JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_createIndexFromTemplate
  (JNIEnv *, jclass, jintArray, jlong, jint, jstring, jbyteArray, jobject);

/*
 * Class:     org_opensearch_knn_jni_FaissService
 * Method:    createBinaryIndexFromTemplate
 * Signature: ([IJILjava/lang/String;[BLjava/util/Map;)V
 */
    JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_createBinaryIndexFromTemplate
      (JNIEnv *, jclass, jintArray, jlong, jint, jstring, jbyteArray, jobject);

/*
 * Class:     org_opensearch_knn_jni_FaissService
 * Method:    createByteIndexFromTemplate
 * Signature: ([IJILjava/lang/String;[BLjava/util/Map;)V
 */
    JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_createByteIndexFromTemplate
      (JNIEnv *, jclass, jintArray, jlong, jint, jstring, jbyteArray, jobject);

/*
 * Class:     org_opensearch_knn_jni_FaissService
 * Method:    loadIndex
 * Signature: (Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_FaissService_loadIndex
  (JNIEnv *, jclass, jstring);

/*
 * Class:     org_opensearch_knn_jni_FaissService
 * Method:    loadBinaryIndex
 * Signature: (Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_FaissService_loadBinaryIndex
  (JNIEnv *, jclass, jstring);

/*
 * Class:     org_opensearch_knn_jni_FaissService
 * Method:    isSharedIndexStateRequired
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_org_opensearch_knn_jni_FaissService_isSharedIndexStateRequired
  (JNIEnv *, jclass, jlong);

/*
 * Class:     org_opensearch_knn_jni_FaissService
 * Method:    initSharedIndexState
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_FaissService_initSharedIndexState
  (JNIEnv *, jclass, jlong);

/*
 * Class:     org_opensearch_knn_jni_FaissService
 * Method:    setSharedIndexState
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_setSharedIndexState
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     org_opensearch_knn_jni_FaissService
 * Method:    queryIndex
 * Signature: (J[FILjava/util/Map[I)[Lorg/opensearch/knn/index/query/KNNQueryResult;
 */
JNIEXPORT jobjectArray JNICALL Java_org_opensearch_knn_jni_FaissService_queryIndex
  (JNIEnv *, jclass, jlong, jfloatArray, jint, jobject, jintArray);

/*
 * Class:     org_opensearch_knn_jni_FaissService
 * Method:    queryIndexWithFilter
 * Signature: (J[FILjava/util/Map[JI[I)[Lorg/opensearch/knn/index/query/KNNQueryResult;
 */
JNIEXPORT jobjectArray JNICALL Java_org_opensearch_knn_jni_FaissService_queryIndexWithFilter
  (JNIEnv *, jclass, jlong, jfloatArray, jint, jobject, jlongArray, jint, jintArray);

/*
 * Class:     org_opensearch_knn_jni_FaissService
 * Method:    queryBIndexWithFilter
 * Signature: (J[BILjava/util/Map[JI[I)[Lorg/opensearch/knn/index/query/KNNQueryResult;
 */
JNIEXPORT jobjectArray JNICALL Java_org_opensearch_knn_jni_FaissService_queryBinaryIndexWithFilter
  (JNIEnv *, jclass, jlong, jbyteArray, jint, jobject, jlongArray, jint, jintArray);

/*
 * Class:     org_opensearch_knn_jni_FaissService
 * Method:    free
 * Signature: (JZ)V
 */
JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_free
  (JNIEnv *, jclass, jlong, jboolean);

/*
 * Class:     org_opensearch_knn_jni_FaissService
 * Method:    freeSharedIndexState
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_freeSharedIndexState
  (JNIEnv *, jclass, jlong);

/*
 * Class:     org_opensearch_knn_jni_FaissService
 * Method:    initLibrary
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_FaissService_initLibrary
  (JNIEnv *, jclass);

/*
 * Class:     org_opensearch_knn_jni_FaissService
 * Method:    trainIndex
 * Signature: (Ljava/util/Map;IJ)[B
 */
JNIEXPORT jbyteArray JNICALL Java_org_opensearch_knn_jni_FaissService_trainIndex
  (JNIEnv *, jclass, jobject, jint, jlong);

/*
 * Class:     org_opensearch_knn_jni_FaissService
 * Method:    trainBinaryIndex
 * Signature: (Ljava/util/Map;IJ)[B
 */
JNIEXPORT jbyteArray JNICALL Java_org_opensearch_knn_jni_FaissService_trainBinaryIndex
  (JNIEnv *, jclass, jobject, jint, jlong);

/*
 * Class:     org_opensearch_knn_jni_FaissService
 * Method:    trainByteIndex
 * Signature: (Ljava/util/Map;IJ)[B
 */
JNIEXPORT jbyteArray JNICALL Java_org_opensearch_knn_jni_FaissService_trainByteIndex
  (JNIEnv *, jclass, jobject, jint, jlong);

/*
 * Class:     org_opensearch_knn_jni_FaissService
 * Method:    transferVectors
 * Signature: (J[[F)J
 */
JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_FaissService_transferVectors
  (JNIEnv *, jclass, jlong, jobjectArray);

/*
* Class:     org_opensearch_knn_jni_FaissService
* Method:    rangeSearchIndexWithFilter
* Signature: (J[FJLjava/util/MapI[JII)[Lorg/opensearch/knn/index/query/RangeQueryResult;
*/
JNIEXPORT jobjectArray JNICALL Java_org_opensearch_knn_jni_FaissService_rangeSearchIndexWithFilter
  (JNIEnv *, jclass, jlong, jfloatArray, jfloat, jobject, jint, jlongArray, jint, jintArray);

/*
 * Class:     org_opensearch_knn_jni_FaissService
 * Method:    rangeSearchIndex
 * Signature: (J[FJLjava/util/MapII)[Lorg/opensearch/knn/index/query/RangeQueryResult;
 */
JNIEXPORT jobjectArray JNICALL Java_org_opensearch_knn_jni_FaissService_rangeSearchIndex
  (JNIEnv *, jclass, jlong, jfloatArray, jfloat, jobject, jint, jintArray);

#ifdef __cplusplus
}
#endif
#endif
