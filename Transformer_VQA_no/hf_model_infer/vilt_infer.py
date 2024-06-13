import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image

import requests
import matplotlib.pyplot as plt

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

url = "https://img0.baidu.com/it/u=1234692956,2844800437&fm=253&fmt=auto&app=138&f=JPEG?w=576&h=386"
image = Image.open(requests.get(url, stream=True).raw)
plt.imshow(image)
plt.show()
text = "What shape is？"

encoding = processor(image, text, return_tensors="pt")
outputs = model(**encoding)
logits = outputs.logits
idx = logits.argmax(-1).item()
print("Question:",text)
print("Predicted answer:", model.config.id2label[idx])
