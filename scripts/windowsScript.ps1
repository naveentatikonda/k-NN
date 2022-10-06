git submodule update --init -- jni/external/nmslib
git submodule update --init -- jni/external/faiss

git config --global user.email navtat@amazon.com
git config --global user.name Naveen Tatikonda
#cat patches/CMakeLists.patch | git am

git apply patches/CMakeLists.patch --verbose

(Get-Content jni/external/faiss/faiss/impl/index_read.cpp).replace('_MSC_VER', '__MINGW32__') | Set-Content jni/external/faiss/faiss/impl/index_read.cpp
(Get-Content jni/external/faiss/faiss/impl/index_write.cpp).replace('_MSC_VER', '__MINGW32__') | Set-Content jni/external/faiss/faiss/impl/index_write.cpp

(Get-Content jni/external/faiss/faiss/OnDiskInvertedLists.cpp).replace('#include <sys/mman.h>', "#ifndef __MINGW32__`n#include <sys/mman.h>`n#endif") | Set-Content jni/external/faiss/faiss/OnDiskInvertedLists.cpp
