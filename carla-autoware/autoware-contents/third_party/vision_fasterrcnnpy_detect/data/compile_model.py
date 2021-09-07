import torch
import torchvision

if __name__ == '__main__':
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True) 
    model.eval()
    # Trace forward operation with example input.
    example_forward_input = torch.rand(1, 3, 224, 224)
    print(model(example_forward_input))
    script_model = torch.jit.script(model)
    script_model.save('fasterrcnn_resnet50.ts')
