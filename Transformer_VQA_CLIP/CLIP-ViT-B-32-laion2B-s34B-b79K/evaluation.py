import torch
from PIL import Image
import open_clip
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
from tqdm import tqdm

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
tokenizer = open_clip.get_tokenizer('ViT-B-32')

TMPLATE = "The image contains {} objects."
ROOT = ""

def predict(image_path, labels):
    image = preprocess(Image.open(image_path)).unsqueeze(0)
    text = tokenizer(labels)

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    return labels[text_probs.argmax(dim=-1).item()]

def evaluate_model(image_paths, true_labels, labels):
    all_preds = []
    
    for image_path in tqdm(image_paths,desc="model eval....."):
        pred_label = predict(image_path, labels)
        all_preds.append(pred_label)
    
    accuracy = accuracy_score(true_labels, all_preds)
    precision = precision_score(true_labels, all_preds, average='weighted', zero_division=1)
    recall = recall_score(true_labels, all_preds, average='weighted', zero_division=1)
    f1 = f1_score(true_labels, all_preds, average='weighted', zero_division=1)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

def load_json(path):
    with open(path, "r", encoding="utf-8") as r:
        data =json.load(r)
    return data

def main(input_file):
    eval_data = load_json(input_file)
    max_label = -1
    true_labels = []
    image_paths = []
    for it in eval_data:
        image_paths.append(os.path.join(ROOT, it["image"]))
        try:
            if int(it["answer"])> max_label:
                max_label = int(it["answer"])
        except:
            true_labels.append(TMPLATE.format(it["answer"]))
            continue

    labels = [TMPLATE.format(i+1) for i in range(max_label)]
    evaluate_model(true_labels, true_labels=true_labels, labels=labels)

main("")
    
