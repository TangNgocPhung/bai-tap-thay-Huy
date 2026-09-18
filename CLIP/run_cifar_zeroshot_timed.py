import os
import time
import clip
import torch
from torchvision.datasets import CIFAR100

device = "cpu"

t0 = time.time()
model, preprocess = clip.load('ViT-B/32', device)
t_load = time.time() - t0

cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

image, class_id = cifar100[3637]
image_input = preprocess(image).unsqueeze(0).to(device)
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

t1 = time.time()
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)

image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
t_infer = time.time() - t1

values, indices = similarity[0].topk(5)

print(f"\nload_time_sec={t_load:.2f}")
print(f"infer_time_sec={t_infer:.3f}")
print(f"Ground truth label: {cifar100.classes[class_id]}\n")
print("Top predictions (openai/CLIP):\n")
for value, index in zip(values, indices):
    print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")
