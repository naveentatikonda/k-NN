#!/bin/bash

# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0

set -ex

# Pull library submodule explicitly. While "cmake ." actually pulls the submodule if its not there, we
# need to pull it before calling cmake. Also, we need to call it from the root git directory.
# Otherwise, the submodule update call may fail on earlier versions of git.
git submodule update --init -- jni/external/nmslib
git submodule update --init -- jni/external/faiss

git apply patches/CMakeLists.patch
sed -i 's/ _MSC_VER/ __MINGW32__/g' jni/external/faiss/faiss/impl/index_read.cpp
sed -i 's/ _MSC_VER/ __MINGW32__/g' jni/external/faiss/faiss/impl/index_write.cpp
sed -i -e 's/#include <sys\/mman.h>/#ifndef __MINGW32__\n#include <sys\/mman.h>\n#endif/' jni/external/faiss/faiss/OnDiskInvertedLists.cpp

git apply patches/libraryName.patch
git apply patches/pluginSecurityPolicy.patch
git apply patches/pluginCoverage.patch