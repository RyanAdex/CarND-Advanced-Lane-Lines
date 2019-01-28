CMAKE_MINIMUM_REQUIRED(VERSION 3.13.2)
# 指定工程名称
PROJECT(Ot)
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_FLAGS  "-std=c++17")
# 导入OpenCV库
# FIND_PACKAGE( OpenCV REQUIRED )
# 设置源文件
aux_source_directory(. DIR_TOOT_SRCS)
# 设置头文件搜索目录
SET( OPENCV_ROOT C:/opencv/build/install/)
INCLUDE_DIRECTORIES(${OPENCV_ROOT}/include/opencv2/ ${OPENCV_ROOT}/include/)
# 设置动态库链接目录
LINK_DIRECTORIES( ${OPENCV_ROOT}/x64/mingw/lib)
# 设置要链接的库文件
SET( LIB_PROJECT 
            opencv_core400
            opencv_highgui400
            opencv_calib3d400
            opencv_imgproc400
            opencv_imgcodecs400
            opencv_videoio400
            opencv_dnn400
            opencv_tracking400)
# 指定执行文件输出目录
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
# 生成可执行文件，即程序
ADD_EXECUTABLE( abc ${PUBLIC_SRC_FILE} ${DIR_TOOT_SRCS}  )
# 要链接的动态库  参数指明库的名字即可，比如libxxxx1.so  指明是xxxx1即可！！！
TARGET_LINK_LIBRARIES( abc ${LIB_PROJECT} ${OpenCV_LIBS} )