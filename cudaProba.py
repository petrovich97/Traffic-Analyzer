
import torch
from torchvision.ops import nms

# Test if torchvision NMS can run with CUDA
#boxes = torch.tensor([[0, 0, 1, 1], [1, 1, 2, 2]], dtype=torch.float32).cuda()
#scores = torch.tensor([0.9, 0.8], dtype=torch.float32).cuda()

#try:
#    result = nms(boxes, scores, 0.5)
#    print("NMS ran successfully on CUDA")
#except Exception as e:

print( torch.__version__)

