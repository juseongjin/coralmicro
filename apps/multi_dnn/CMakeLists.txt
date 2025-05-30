# Multi-DNN FreeRTOS 프로젝트용 CMakeLists.txt
# coralmicro/apps/multi_dnn/ 디렉토리에 위치

set(multi_dnn_SOURCES 
    main.cc
)

set(multi_dnn_LINK_LIBRARIES
    libs_base-m7_freertos
    libs_jpeg_m7
    libs_tensorflow-m7
)

add_executable_m7(multi_dnn
    ${multi_dnn_SOURCES}
    DATA
    # ${PROJECT_SOURCE_DIR}/models/person_detect_model.tflite
    # ${PROJECT_SOURCE_DIR}/models/trained_lstm_int8.tflite
    ${PROJECT_SOURCE_DIR}/models/ssd_mobilenet_v1_coco_quant_no_nms.tflite
    # ${PROJECT_SOURCE_DIR}/models/mcunet-320kb-1mb_imagenet.tflite
    ${PROJECT_SOURCE_DIR}/models/person.raw
)

target_link_libraries(multi_dnn
    ${multi_dnn_LINK_LIBRARIES}
)