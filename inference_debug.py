import mmcv

from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
register_all_modules()

config1 = "configs/efficient_net.py"
checkpoint1 = "checkpoints/efficient_net/epoch_500.pth"

model = init_detector(config1, checkpoint1, device='cuda:0')
img = mmcv.imread('demo.jpg', channel_order='rgb')
result = inference_detector(model, img)
print(result)

pass

"""
config2 = "configs/fcenet.py"
checkpoint2 = "checkpoints/fcenet/epoch_400.pth"


device='cuda:0'


config2 = mmcv.Config.fromfile(config2)
config2.model.pretrained = None


model2 = build_detector(config2.model)

# Load checkpoint
checkpoint = load_checkpoint(model, checkpoint, map_location=device)

# Set the classes of models for inference
model.CLASSES = checkpoint['meta']['CLASSES']

# We need to set the model's cfg for inference
model.cfg = config

# Convert the model to GPU
model.to(device)
# Convert the model into evaluation mode
model.eval()
"""