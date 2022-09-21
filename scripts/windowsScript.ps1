#
# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#

git submodule update --init -- jni/external/nmslib
git submodule update --init -- jni/external/faiss

git apply patches/CMakeLists.patch --verbose

# Validating if the CMakeLists patch is applied
type jni/CMakeLists.txt | Select-String "Windows"

(Get-Content jni/external/faiss/faiss/impl/index_read.cpp).replace('_MSC_VER', '__MINGW32__') | Set-Content jni/external/faiss/faiss/impl/index_read.cpp
(Get-Content jni/external/faiss/faiss/impl/index_write.cpp).replace('_MSC_VER', '__MINGW32__') | Set-Content jni/external/faiss/faiss/impl/index_write.cpp

(Get-Content jni/external/faiss/faiss/OnDiskInvertedLists.cpp).replace('#include <sys/mman.h>', "#ifndef __MINGW32__`n#include <sys/mman.h>`n#endif") | Set-Content jni/external/faiss/faiss/OnDiskInvertedLists.cpp
