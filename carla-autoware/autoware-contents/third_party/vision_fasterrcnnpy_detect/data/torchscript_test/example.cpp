#include <torch/script.h>
#include <torch/torch.h>
#include <torchvision/vision.h>
#include <torchvision/ops/nms.h>
#include <iostream>
#include <typeinfo>


torch::jit::script::Module loadModel(const std::string& path) {
  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(path);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
  }

  std::cout << "loading ok\n";
  return module;
}


auto runInferenceCPU(torch::jit::script::Module& module) {
  // TorchScript models require a List[IValue] as input
  std::vector<torch::jit::IValue> inputs;

  // Faster RCNN accepts a List[Tensor] as main input
  std::vector<torch::Tensor> images;
  images.push_back(torch::rand({3, 256, 275}));
  images.push_back(torch::rand({3, 256, 275}));

  inputs.push_back(images);
  auto output = module.forward(inputs);

  std::cout << typeid(output).name() << '\n';

  std::cout << "CPU inference ok\n";
  std::cout << "output" << output << "\n";
  return output;
}

void runInferenceGPU(torch::jit::script::Module& module) {
  // TorchScript models require a List[IValue] as input
  std::vector<torch::jit::IValue> inputs;

  // Faster RCNN accepts a List[Tensor] as main input
  std::vector<torch::Tensor> images;

  if (torch::cuda::is_available()) {
    // Move traced model to GPU
    module.to(torch::kCUDA);

    // Add GPU inputs
    images.clear();
    inputs.clear();

    torch::TensorOptions options = torch::TensorOptions{torch::kCUDA};
    images.push_back(torch::rand({3, 256, 275}, options));
    images.push_back(torch::rand({3, 256, 275}, options));

    inputs.push_back(images);
    auto output = module.forward(inputs);

    std::cout << "GPU inference ok\n";
    std::cout << "output" << output << "\n";
    return;
  }
  std::cout << "GPU inference not ok\n";
}

int main() {
  auto module = loadModel("./model.ts");
  auto result = runInferenceCPU(module);
  std::cout << result << std::endl;
  std::cout << result.toTuple()->elements()[1]<< std::endl;
  auto detections = result.toTuple()->elements()[1].toList().get(0).toGenericDict();
  auto boxes = detections.at("boxes").toList();
  std::cout << "Boxes: " << detections.at("boxes") << std::endl;
  auto scores = detections.at("scores").toList();
  auto labels = detections.at("labels").toList();
  std::cout << boxes.size() << scores.size() << labels.size() << std::endl;
  //std:cout << detections << std::endl;
  //std::cout << *detections.type() << std::endl;
  //std::cout << detections.at("boxes") << std::endl;
  //for (auto detection: detections.at("boxes")) {
  //  std::cout << detection.key() << std::endl;
  //}
  runInferenceGPU(module);
}

