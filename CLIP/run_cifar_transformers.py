import os
import time
import torch
from torchvision.datasets import CIFAR100
from transformers import CLIPModel, CLIPProcessor

cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
image, class_id = cifar100[3637]

t0 = time.time()
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
t_load = time.time() - t0

model.eval()

prompts = [f"a photo of a {c}" for c in cifar100.classes]

t1 = time.time()
inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
with torch.no_grad():
    outputs = model(**inputs)
t_infer = time.time() - t1

probs = outputs.logits_per_image.softmax(dim=1)[0]
values, indices = probs.topk(5)

print(f"\nload_time_sec={t_load:.2f}")
print(f"infer_time_sec={t_infer:.3f}")
print(f"Ground truth label: {cifar100.classes[class_id]}\n")
print("Top predictions (transformers / openai/clip-vit-base-patch32):\n")
for value, index in zip(values, indices):
    print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")
