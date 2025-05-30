// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0

#include "libs/base/filesystem.h"
#include "libs/base/led.h"
#include "libs/base/strings.h"
#include "libs/base/tasks.h"
#include "libs/base/timer.h"

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

// Runs face detection on the Edge TPU, using the on-board camera, printing
// results to the serial console and turning on the User LED when a face
// is detected.
//
// Change model load, resolver
// Change CMakeLists.txt model

namespace coralmicro {
namespace {

constexpr int kPersonIndex = 1;
constexpr int kNotAPersonIndex = 0;
constexpr int kTensorArenaSizePerson = 1024 * 1024;
constexpr int kTensorArenaSizeLstm = 1024 * 1024;
STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena_person, kTensorArenaSizePerson);
STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena_lstm, kTensorArenaSizeLstm);


[[noreturn]] void PersonDetectTask(void* param) {
  vTaskDelay(pdMS_TO_TICKS(1000));

  /////////// Load Model ///////////
  std::vector<uint8_t> model;
  if (!LfsReadFile("/models/ssd_mobilenet_v1_coco_quant_no_nms.tflite", &model)) {
  // if (!LfsReadFile("/models/mcunet-320kb-1mb_imagenet.tflite", &model)) {
  // if (!LfsReadFile("/models/person_detect_model.tflite", &model)) {
    printf("Error: Failed to load person_detect_model\r\n");
    vTaskSuspend(nullptr);
  }
  printf("Model load done\r\n");
  fflush(stdout);
  //////////////////////////////////

  /////////// Interprete ///////////
  tflite::MicroErrorReporter error_reporter;
  // tflite::MicroMutableOpResolver<5> resolver;
  // resolver.AddAveragePool2D();
  // resolver.AddConv2D();
  // resolver.AddDepthwiseConv2D();
  // resolver.AddReshape();
  // resolver.AddSoftmax();
  // mcunet
  // tflite::MicroMutableOpResolver<6> resolver;
  // resolver.AddConv2D();
  // resolver.AddPad();
  // resolver.AddDepthwiseConv2D();
  // resolver.AddReshape();
  // resolver.AddAdd();
  // resolver.AddAveragePool2D();
  // ssd
  tflite::MicroMutableOpResolver<6> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddReshape();
  resolver.AddConcatenation();
  resolver.AddLogistic();
  resolver.AddDetectionPostprocess();
  tflite::MicroInterpreter interpreter(tflite::GetModel(model.data()), resolver,
                                       tensor_arena_person, kTensorArenaSizePerson, &error_reporter);

  //////////////////////////////////

  //////// AllocateTensor /////////
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
  //////////////////////////////////

  /////////// Inference ////////////
  std::vector<float> time;
  int n = 0;
  // while (true) {
  while (n < 30) {
    float start = static_cast<float>(TimerMicros());
    if (interpreter.Invoke() != kTfLiteOk) {
      printf("Person: Inference failed\r\n");
    } else {
      auto* output = interpreter.output(0);
      // Process the inference results.
      // auto person_score =
      // static_cast<int8_t>(output->data.uint8[kPersonIndex]);
      // auto no_person_score =
      // static_cast<int8_t>(output->data.uint8[kNotAPersonIndex]);
      // time.push_back((static_cast<float>(TimerMicros()) - start)/1000);
      // printf("Task 1 %dth latency: %.6f ms (score: %d)\r\n", n+1, (static_cast<float>(TimerMicros()) - start)/1000, person_score);
      printf("Task 1 %dth latency: %.6f ms\r\n", n+1, (static_cast<float>(TimerMicros()) - start)/1000);
      fflush(stdout);
    }
    // vTaskDelay(pdMS_TO_TICKS(1));
    taskYIELD();
    n++;
  }
  // for(int i=0; i < time.size(); i++){
  //   printf("Task 1 %dth latency: %.6f ms\r\n", i+1, time[i]);
  //   fflush(stdout);
  // }
  vTaskDelete(NULL);
  //////////////////////////////////
}

[[noreturn]] void LstmTask(void* param) {
  vTaskDelay(pdMS_TO_TICKS(1000));

  /////////// Load Model ////////////
  std::vector<uint8_t> model;
  if (!LfsReadFile("/models/trained_lstm_int8.tflite", &model)) {
      printf("Error: Failed to load LSTM model\r\n");
    vTaskSuspend(nullptr);
  }
  printf("Model load done\r\n");
  fflush(stdout);
  //////////////////////////////////

  //////////// Interprete //////////
  tflite::MicroErrorReporter error_reporter;
  // LSTM
  tflite::MicroMutableOpResolver<4> resolver;
  resolver.AddUnidirectionalSequenceLSTM();
  resolver.AddReshape();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
 
  tflite::MicroInterpreter interpreter(tflite::GetModel(model.data()), resolver,
                                       tensor_arena_lstm, kTensorArenaSizeLstm, &error_reporter);
  //////////////////////////////////

  ///////// AllocateTensor /////////
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
  //////////////////////////////////

  //////////// Inference ///////////
  std::vector<float> time;
  int n = 0;
  // while (true) {
  while (n < 100) {
    float start = static_cast<float>(TimerMicros());
    if (interpreter.Invoke() != kTfLiteOk) {
      printf("LSTM: Inference failed\r\n");
    } else {
      time.push_back((static_cast<float>(TimerMicros()) - start)/1000);
      // printf("Task 2 %dth latency: %.6f ms\r\n", n, (static_cast<float>(TimerMicros()) - start)/1000);
      // fflush(stdout);
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
    }
    // vTaskDelay(pdMS_TO_TICKS(1));
    taskYIELD();
    n++;
  }
  for(int i=0; i < time.size(); i++){
    printf("Task 2 %dth latency: %.6f ms\r\n", i+1, time[i]);
    fflush(stdout);
  }
  vTaskDelete(NULL);
  ////////////////////////////////////
}

[[noreturn]] void Main() {
  // Task 시작 부분
  LedSet(Led::kUser, 1);  // User LED ON

  xTaskCreate(PersonDetectTask, "person_detect", configMINIMAL_STACK_SIZE * 30, nullptr, kAppTaskPriority, nullptr);
  // xTaskCreate(LstmTask, "lstm_task", configMINIMAL_STACK_SIZE * 30, nullptr, kAppTaskPriority, nullptr);
  vTaskStartScheduler();
}

}  // namespace
}  // namespace coralmicro

extern "C" void app_main(void* param) {
  (void)param;
  coralmicro::Main();
}
