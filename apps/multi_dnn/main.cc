// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0

#include "libs/base/filesystem.h"
#include "libs/base/led.h"
#include "libs/base/strings.h"
#include "libs/base/tasks.h"

#include "libs/tensorflow/utils.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"
#include "third_party/tflite-micro/tensorflow/lite/schema/schema_generated.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/all_ops_resolver.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_error_reporter.h"

#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"

#include <cstdio>
#include <vector>
#include <cstring>

inline void DWT_Init() {
  if (!(CoreDebug->DEMCR & CoreDebug_DEMCR_TRCENA_Msk)) {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  }
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

inline uint32_t micros_dwt() {
  return DWT->CYCCNT / (SystemCoreClock / 1000000);
}

namespace coralmicro {
namespace {

constexpr int kPersonIndex = 1;
constexpr int kNotAPersonIndex = 0;
constexpr int kTensorArenaSizePerson = 100 * 1024;
constexpr int kTensorArenaSizeLstm = 10 * 1024;
STATIC_TENSOR_ARENA_IN_OCRAM(tensor_arena_person, kTensorArenaSizePerson);
STATIC_TENSOR_ARENA_IN_OCRAM(tensor_arena_lstm, kTensorArenaSizeLstm);


[[noreturn]] void PersonDetectTask(void* param) {
  std::vector<uint8_t> model;
    if (!LfsReadFile("/models/person_detect_model.tflite", &model)) {
    printf("Error: Failed to load person_detect_model\r\n");
    vTaskSuspend(nullptr);
  }
  printf("Model load done\r\n");

  tflite::MicroErrorReporter error_reporter;
  tflite::MicroMutableOpResolver<5> resolver;
  resolver.AddAveragePool2D();
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddReshape();
  resolver.AddSoftmax();
  tflite::MicroInterpreter interpreter(tflite::GetModel(model.data()), resolver,
                                       tensor_arena_person, sizeof(tensor_arena_person), &error_reporter);

  if (interpreter.AllocateTensors() != kTfLiteOk) {
    printf("Person: AllocateTensors() failed\r\n");
    vTaskSuspend(nullptr);
  }

  auto* input = interpreter.input_tensor(0);

  if (!LfsReadFile("/models/person.raw", tflite::GetTensorData<uint8_t>(input),
                    input->bytes)) {
    printf("Person: Failed to load input image\r\n");
    vTaskSuspend(nullptr);
  }

  DWT_Init(); // 최초 1회 초기화
  int n = 1;
  uint32_t start_us, end_us;
  float elapsed_ms;
  while (true) {
    start_us = micros_dwt();
    if (interpreter.Invoke() != kTfLiteOk) {
      printf("Person: Inference failed\r\n");
    } else {
      end_us = micros_dwt();
      elapsed_ms = (float)(end_us - start_us) / 1000;
      auto* output = interpreter.output(0);
      // Process the inference results.
      auto person_score =
      static_cast<int8_t>(output->data.uint8[kPersonIndex]);
      auto no_person_score =
      static_cast<int8_t>(output->data.uint8[kNotAPersonIndex]);
      printf("%d's [latency, score]: [%.6fms, %d]\r\n", n++, elapsed_ms, person_score);
    }
    vTaskDelay(pdMS_TO_TICKS(1000));
  }
}

[[noreturn]] void LstmTask(void* param) {
  // tensor_arena_lstm = (uint8_t*)malloc(kTensorArenaSizeLstm);
  // if (!tensor_arena_lstm) {
  //   printf("LSTM: Failed to allocate tensor arena memory\r\n");
  //   vTaskSuspend(nullptr);
  // }

  std::vector<uint8_t> model;
  if (!LfsReadFile("/models/trained_lstm_int8.tflite", &model)) {
  // if (!LfsReadFile("/models/ssd_mobilenet_v1_coco_quant_no_nms.tflite", &model)) {
    printf("Error: Failed to load LSTM model\r\n");
    fflush(stdout);
    vTaskSuspend(nullptr);
  }
  printf("Model load done\r\n");
  tflite::MicroErrorReporter error_reporter;
  // LSTM
  tflite::MicroMutableOpResolver<4> resolver;
  resolver.AddUnidirectionalSequenceLSTM();
  resolver.AddReshape();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  // mcunet
  // tflite::MicroMutableOpResolver<6> resolver;
  // resolver.AddConv2D();
  // resolver.AddPad();
  // resolver.AddDepthwiseConv2D();
  // resolver.AddReshape();
  // resolver.AddAdd();
  // resolver.AddAveragePool2D();
  // ssd
  // tflite::MicroMutableOpResolver<6> resolver;
  // resolver.AddConv2D();
  // resolver.AddDepthwiseConv2D();
  // resolver.AddReshape();
  // resolver.AddConcatenation();
  // resolver.AddLogistic();
  // resolver.AddDetectionPostprocess();

  tflite::MicroInterpreter interpreter(tflite::GetModel(model.data()), resolver,
                                       tensor_arena_lstm, sizeof(tensor_arena_lstm), &error_reporter);

  if (interpreter.AllocateTensors() != kTfLiteOk) {
    printf("LSTM: AllocateTensors() failed\r\n");
        vTaskSuspend(nullptr);
  }
  
  TfLiteTensor* input = interpreter.input(0);

  std::vector<uint8_t> input_data(input->bytes, 0x1);  // 예시로 중간값(64)으로 채움
  memcpy(input->data.uint8, input_data.data(), input->bytes);
  
  // if (!LfsReadFile("/models/person.raw", tflite::GetTensorData<uint8_t>(input),
  //                   input->bytes)) {
  //   printf("SSD: Failed to load input image\r\n");
  //   vTaskSuspend(nullptr);
  // }


  DWT_Init(); // 최초 1회 초기화
  int n = 1;
  uint32_t start_us, end_us;
  float elapsed_ms;
  while (true) {
    start_us = micros_dwt();
    if (interpreter.Invoke() != kTfLiteOk) {
      printf("LSTM: Inference failed\r\n");
    } else {
      end_us = micros_dwt();
      elapsed_ms = (float)(end_us - start_us) / 1000;
      TfLiteTensor* output = interpreter.output(0);

      // printf("%d's [latency, result]: [%.6fms, %d]\r\n", n++, elapsed_ms, output->data.uint8[0]);
      // int8_t* output_data = output->data.int8;
      // int output_size = output->bytes; // 보통 1000 바이트, 즉 1000개 클래스

      // 예) 가장 높은 점수(최댓값) 찾기
      // int max_index = 0;
      // int8_t max_value = output_data[0];
      // for (int i = 1; i < output_size; i++) {
      //     if (output_data[i] > max_value) {
      //         max_value = output_data[i];
      //         max_index = i;
      //     }
      // }
      printf("%d's [latency]: [%.6fms]\r\n", n++, elapsed_ms);
    }
    vTaskDelay(pdMS_TO_TICKS(1000));
  }
}

[[noreturn]] void Main() {
  // Task 시작 부분
  LedSet(Led::kUser, 1);  // 예: 빨간 LED ON

  xTaskCreate(PersonDetectTask, "person_detect", configMINIMAL_STACK_SIZE * 30, nullptr, kAppTaskPriority, nullptr);
  xTaskCreate(LstmTask, "lstm_task", configMINIMAL_STACK_SIZE * 30, nullptr, kAppTaskPriority, nullptr);
  vTaskSuspend(nullptr);
}

}  // namespace
}  // namespace coralmicro

extern "C" void app_main(void* param) {
  (void)param;
  coralmicro::Main();
}
