

#   ---[ Google-gflags
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindGFlags.cmake)
list(APPEND Samples_INCLUDE_DIRS ${GFLAGS_INCLUDE_DIRS})
list(APPEND Samples_LINK_LIBS ${GFLAGS_LIBRARIES})
# message("111111111111",${Samples_LINK_LIBS})
# ---[ OpenCV
set(OpenCV_FIND_QUIETLY true)
# find_package(OpenCV OPTIONAL_COMPONENTS imgcodecs)
# message("111111111111",${OPTIONAL_COMPONENTS})
# set(imgcodecs_libs ${OpenCV_LIBS})
# message("111111111111",${OpenCV_LIBS})
#find_package(OpenCV REQUIRED core imgproc highgui)
#if (OpenCV_FOUND)
#  message(STATUS "OpenCV Found")
#else (OpenCV_FOUND)
#  message(FATAL_ERROR "Can not find OpenCV libs!")
#endif ()
#if (imgcodecs_FOUND)
#  list(APPEND OpenCV_LIBS ${imgcodecs_libs})
#endif ()
add_compile_options("-g")
set(OpenCV_lib 
  /data1/wcl/mlu/174/wcl/usr/usr/libs/opencv_400/libopencv_imgproc.so
  /data1/wcl/mlu/174/wcl/usr/usr/libs/opencv_400/libopencv_highgui.so
  /data1/wcl/mlu/174/wcl/usr/usr/libs/opencv_400/libopencv_core.so
  /data1/wcl/mlu/174/wcl/usr/usr/libs/opencv_400/libopencv_imgcodecs.so
  /data1/wcl/mlu/174/wcl/usr/usr/libs/opencv_400/libopencv_videoio.so
  /data1/wcl/mlu/174/wcl/usr/usr/libs/opencv_400/libopencv_video.so
  /data1/wcl/mlu/174/wcl/usr/usr/libs/opencv_400/libopencv_dnn.so
  /data1/wcl/mlu/174/wcl/usr/usr/libs/opencv_400/libopencv_tracking.so
  /data1/wcl/mlu/174/wcl/usr/usr/libs/opencv_400/libopencv_calib3d.so
  /data1/wcl/mlu/174/wcl/usr/usr/libs/opencv_400/libopencv_features2d.so
  /data1/wcl/mlu/174/wcl/usr/usr/libs/opencv_400/libopencv_shape.so
  /data1/zhn/soft/json/libjsoncpp.so.1
  # /data/wcl/174/wcl/usr/usr/libs/opencv_400/libopencv_imgproc.so
  # /data/wcl/174/wcl/usr/usr/libs/opencv_400/libopencv_highgui.so
  # /data/wcl/174/wcl/usr/usr/libs/opencv_400/libopencv_core.so
  # /data1/wcl/usr/opencv/lib/libopencv_imgproc.so
  # /data1/wcl/usr/opencv/lib/libopencv_highgui.so
  # /data1/wcl/usr/opencv/lib/libopencv_core.so

  )
set(OpenCV_include 
  /data1/wcl/mlu/174/wcl/usr/usr/include/opencv2
  /data1/wcl/mlu/174/wcl/usr/usr/include
  /data1/wcl/mlu/174/wcl/usr/usr/include/opencv
  /data1/zhn/soft/json/
# /data/wcl/174/wcl/usr/usr/include/opencv2
# /data/wcl/174/wcl/usr/usr/include/
# /data/wcl/174/wcl/usr/usr/include/opencv 
# /data1/wcl/usr/opencv/include/opencv4
)

# list(APPEND Samples_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
# list(APPEND Samples_LINK_LIBS ${OpenCV_LIBS})

list(APPEND Samples_INCLUDE_DIRS ${OpenCV_include})
list(APPEND Samples_LINK_LIBS ${OpenCV_lib})
# message("222222222222",${Samples_LINK_LIBS})
# message("222222222222",${Samples_INCLUDE_DIRS})
# ---[ thread
list(APPEND Samples_LINK_LIBS pthread dl)
# message("333333333333",${dl})

file(GLOB_RECURSE SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
set(COM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include")

include_directories(${Samples_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/include /usr/local/neuware/include/)

add_library(cnrtinfer SHARED  ${SOURCE_FILES} )


target_link_libraries(cnrtinfer easydk ${Samples_LINK_LIBS})


# message("5555555555",${Samples_LINK_LIBS})
install(TARGETS cnrtinfer
DESTINATION lib)
install(DIRECTORY include
DESTINATION include)

add_subdirectory(test)

