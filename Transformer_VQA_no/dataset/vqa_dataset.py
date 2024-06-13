import os
import json
import random
from PIL import Image
from torch.utils.data import Dataset
from dataset.utils import pre_question


class vqa_dataset(Dataset):
    def __init__(self, ann_file, transform, data_root, eos='[SEP]', split="train", max_ques_words=30, answer_list=''):
        self.split = split        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))

        self.transform = transform

        self.data_root = data_root
        self.max_ques_words = max_ques_words
        self.eos = eos
        
        if split=='test':
            self.max_ques_words = 50 # do not limit question length during test
            self.answer_list = json.load(open(answer_list,'r'))    
                
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        image_path = os.path.join(self.data_root,ann['image'])

            
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          
        
        if self.split == 'test':

            question = pre_question(ann['question'], self.max_ques_words)

            if ann['dataset']=='pre':
                label = ann['answer']
                return image_path, image, question, label
            elif ann['dataset']=='vqa':
                question_id = ann['question_id']
                answers = ann['answer']
                # answers = [answer+self.eos for answer in answers]
                return image, question, question_id,answers

        elif self.split=='train':                       
            
            question = pre_question(ann['question'],self.max_ques_words)        
            


            if ann['dataset']=='pre':
                answers = [ann['answer']]
                weights = [0.5]
            elif ann['dataset']=='vqa':
                
                answer_weight = {}
                for answer in ann['answer']:
                    if answer in answer_weight.keys():
                        answer_weight[answer] += 1/len(ann['answer'])
                    else:
                        answer_weight[answer] = 1/len(ann['answer'])

                answers = list(answer_weight.keys())
                weights = list(answer_weight.values())



            answers = [answer+self.eos for answer in answers]
                
            return image, question, answers, weights