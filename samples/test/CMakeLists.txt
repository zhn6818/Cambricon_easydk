cmake_minimum_required(VERSION 2.8)
project(cnrt)

set(ffmpeglib     
/data1/wcl/mlu/174/wcl/usr/usr/libs/ffmpeg/libavcodec.so.57
/data1/wcl/mlu/174/wcl/usr/usr/libs/ffmpeg/libavformat.so.57
/data1/wcl/mlu/174/wcl/usr/usr/libs/ffmpeg/libavutil.so.55
/data1/wcl/mlu/174/wcl/usr/usr/libs/ffmpeg/libswscale.so.4
)
list(APPEND Samples_LINK_LIBS ${ffmpeglib})
add_executable(classification classification.cpp)
target_link_libraries(classification  
    cnrtinfer
    ${Samples_LINK_LIBS}
)


add_executable(insegment insegment.cpp)
target_link_libraries(insegment  
    cnrtinfer
    ${Samples_LINK_LIBS}
)

add_executable(stream_app stream_app.cpp)
target_link_libraries(stream_app  
    cnrtinfer
    ${Samples_LINK_LIBS}
)

add_executable(crowdcount crowdcount.cpp)
target_link_libraries(crowdcount  
    cnrtinfer
    ${Samples_LINK_LIBS}
)



add_executable(detection detection.cpp)
target_link_libraries(detection  
    cnrtinfer
    ${Samples_LINK_LIBS}
)

add_executable(testclass testClassify.cpp)
target_link_libraries(testclass  
    cnrtinfer
    ${Samples_LINK_LIBS}
)

add_executable(test_segment test_segment.cpp)
target_link_libraries(test_segment  
    cnrtinfer
    ${Samples_LINK_LIBS}
)

add_executable(detection_batch detection_batch.cpp)
target_link_libraries(detection_batch  
    cnrtinfer
    ${Samples_LINK_LIBS}
)

add_executable(testTiny testTiny.cpp)
target_link_libraries(testTiny  
    cnrtinfer
    ${Samples_LINK_LIBS}
)

# add_executable(testMtcnn testMtcnn.cpp)
# target_link_libraries(testMtcnn  
#    cnrtinfer
#    ${Samples_LINK_LIBS}
# )

add_executable(test_fpnseg test_fpnseg.cpp)
target_link_libraries(test_fpnseg  
    cnrtinfer
    ${Samples_LINK_LIBS}
)
add_executable(test_HDdetect testHDdetect.cpp)
target_link_libraries(test_HDdetect  
    cnrtinfer
    ${Samples_LINK_LIBS}
)
# add_executable(test_pose test_pose.cpp)
# target_link_libraries(test_pose  
#     cnrtinfer
#     ${Samples_LINK_LIBS}
# )
add_executable(test_yolov4csp test_yolov4csp.cpp)
target_link_libraries(test_yolov4csp  
    cnrtinfer
    ${Samples_LINK_LIBS}
)

add_executable(test_DDRnet test_DDRnet.cpp)
target_link_libraries(test_DDRnet  
    cnrtinfer
    ${Samples_LINK_LIBS}
)
add_executable(test_road test_road.cpp)
target_link_libraries(test_road  
    cnrtinfer
    ${Samples_LINK_LIBS}
)