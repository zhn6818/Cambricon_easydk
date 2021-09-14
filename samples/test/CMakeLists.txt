cmake_minimum_required(VERSION 2.8)
project(cnrt)


add_executable(classification classification.cpp)
target_link_libraries(classification  
    cnrtinfer
    ${Samples_LINK_LIBS}
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
    /data1/wcl/mlu/174/wcl/usr/usr/libs/ffmpeg/libavcodec.so.57
    /data1/wcl/mlu/174/wcl/usr/usr/libs/ffmpeg/libavformat.so.57
    /data1/wcl/mlu/174/wcl/usr/usr/libs/ffmpeg/libavutil.so.55
    /data1/wcl/mlu/174/wcl/usr/usr/libs/ffmpeg/libswscale.so.4
)


add_executable(insegment insegment.cpp)
target_link_libraries(insegment  
    cnrtinfer
    ${Samples_LINK_LIBS}
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
    /data1/wcl/mlu/174/wcl/usr/usr/libs/ffmpeg/libavcodec.so.57
    /data1/wcl/mlu/174/wcl/usr/usr/libs/ffmpeg/libavformat.so.57
    /data1/wcl/mlu/174/wcl/usr/usr/libs/ffmpeg/libavutil.so.55
    /data1/wcl/mlu/174/wcl/usr/usr/libs/ffmpeg/libswscale.so.4
)

add_executable(stream_app stream_app.cpp)
target_link_libraries(stream_app  
    cnrtinfer
    ${Samples_LINK_LIBS}
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
    /data1/wcl/mlu/174/wcl/usr/usr/libs/ffmpeg/libavcodec.so.57
    /data1/wcl/mlu/174/wcl/usr/usr/libs/ffmpeg/libavformat.so.57
    /data1/wcl/mlu/174/wcl/usr/usr/libs/ffmpeg/libavutil.so.55
    /data1/wcl/mlu/174/wcl/usr/usr/libs/ffmpeg/libswscale.so.4
)

add_executable(crowdcount crowdcount.cpp)
target_link_libraries(crowdcount  
    cnrtinfer
    ${Samples_LINK_LIBS}
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
    /data1/wcl/mlu/174/wcl/usr/usr/libs/ffmpeg/libavcodec.so.57
    /data1/wcl/mlu/174/wcl/usr/usr/libs/ffmpeg/libavformat.so.57
    /data1/wcl/mlu/174/wcl/usr/usr/libs/ffmpeg/libavutil.so.55
    /data1/wcl/mlu/174/wcl/usr/usr/libs/ffmpeg/libswscale.so.4
)



add_executable(detection detection.cpp)
target_link_libraries(detection  
    cnrtinfer
    ${Samples_LINK_LIBS}
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
    /data1/wcl/mlu/174/wcl/usr/usr/libs/ffmpeg/libavcodec.so.57
    /data1/wcl/mlu/174/wcl/usr/usr/libs/ffmpeg/libavformat.so.57
    /data1/wcl/mlu/174/wcl/usr/usr/libs/ffmpeg/libavutil.so.55
    /data1/wcl/mlu/174/wcl/usr/usr/libs/ffmpeg/libswscale.so.4
)

add_executable(testclass testClassify.cpp)
target_link_libraries(testclass  
    cnrtinfer
    ${Samples_LINK_LIBS}
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
    /data1/wcl/mlu/174/wcl/usr/usr/libs/ffmpeg/libavcodec.so.57
    /data1/wcl/mlu/174/wcl/usr/usr/libs/ffmpeg/libavformat.so.57
    /data1/wcl/mlu/174/wcl/usr/usr/libs/ffmpeg/libavutil.so.55
    /data1/wcl/mlu/174/wcl/usr/usr/libs/ffmpeg/libswscale.so.4
)