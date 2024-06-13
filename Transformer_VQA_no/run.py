import argparse
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.chdir("/home/admin1/5703-upload/5703/Transformer_VQA_no")
# os.environ["WANDB_DEBUG"] = "true"
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from src.pre_vqa import PreVQA
from src.vision_transformer import interpolate_pos_embed
from src.tokenization_bert import BertTokenizer

import utils
from dataset.utils import save_result
from dataset import create_dataset, create_sampler, create_loader, vqa_collate_fn
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

from scheduler import create_scheduler
from optim import create_optimizer
from sklearn import metrics

os.environ['CUDA_VISIBLE_DEVICES']="0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# sweep_config = {
#     "method": "random",
#     "metric": {"goal": "minimize", "name": "loss"},
#     "parameters": {
#         # "x": {"max": 0.1, "min": 0.01},
#         # "y": {"values": [1, 3, 7]},
#         "batch_size_train":{"values": [8, 16, 32]},
#         "batch_size_test":{"values": [8, 16, 32]},
#         "optimizer.opt": {"values": ["adamW", "RMSprop", "Lookahead"]},
#         "scheduler.sched": {"values": ["cosine", "ReduceLROnPlateau", "StepLR"]}
#     },
# }

# sweep_id = wandb.sweep(sweep=sweep_config, project="my-first-sweep")

run = wandb.init(
    # project="vqa2",
    project="my-first-sweep",
    # Track hyperparameters and run metadata
    config={
        "config": 'config',
    },
)


def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50    
    step_size = 100
    warmup_iterations = warmup_steps*step_size  
    
    #50个batch打印一次loss
    total_loss = 0
    for i,(image, question, answer, weights, n) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image, weights = image.to(device,non_blocking=True), weights.to(device,non_blocking=True)      
        question_input = tokenizer(question, padding='longest', truncation=True, max_length=25, return_tensors="pt").to(device) 
        answer_input = tokenizer(answer, padding='longest', return_tensors="pt").to(device) 
        
        if epoch>0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1, i / len(data_loader))

        loss = model(image, question_input, answer_input, train=True, alpha=alpha, k=n, weights=weights)        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        total_loss+=loss.item()
        if(i!=0 and i%25==0):
            loss_ = total_loss/25
            wandb.log(
                {
                    'loss':loss_
                }
            )
            total_loss = 0
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size) 

        #wandb.log(
        #    {
        #        'learning_rate': optimizer.param_groups[0]["lr"]
        #    }
        #)
            
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()} 


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config) :
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 50
    
    result = []
    
    answer_list = [answer+config['eos'] for answer in data_loader.dataset.answer_list]
    answer_input = tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)    
        
    # for n, (image, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    #     image = image.to(device,non_blocking=True)
    #     question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)
    #
    #     topk_ids, topk_probs = model(image, question_input, answer_input, train=False, k=config['k_test'])
    #
    #     for ques_id, topk_id, topk_prob in zip(question_id, topk_ids, topk_probs):
    #         ques_id = int(ques_id.item())
    #         _, pred = topk_prob.max(dim=0)
    #         result.append({"question_id":ques_id, "answer":data_loader.dataset.answer_list[topk_id[pred]]})

    total = 0
    acc = 0
    gt_labels = []
    pred_labels = []
    for n, (image_path, image, question, label) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)
        question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)

        topk_ids, topk_probs = model(image, question_input, answer_input, train=False, k=config['k_test'])

        for path, label_id, topk_id, topk_prob in zip(image_path, label, topk_ids, topk_probs):
            #ques_id = int(ques_id.item())
            total += 1
            gt_labels.append(str(label_id))
            _, pred = topk_prob.max(dim=0)
            result.append({"answer": label_id, "pred": data_loader.dataset.answer_list[topk_id[pred]]})
            pred = data_loader.dataset.answer_list[topk_id[pred]]
            pred_labels.append(pred)
            # print(str(label_id))
            # print(str(pred))
            if str(label_id) == pred:
                acc += 1 
            
            # print ({"image": path, "answer": label_id, "pred":pred })
    accuracy = acc / total if total > 0 else 0.0
    metric = metrics.classification_report(gt_labels, pred_labels,output_dict=True)
    macro_metric = {
         'macro_prec': metric['macro avg']['precision'],
         'macro_recall': metric['macro avg']['recall'],
         'macro_f1': metric['macro avg']['f1-score']
     }
    weighted_metric = {
         'weighted_prec': metric['weighted avg']['precision'],
         'weighted_recall': metric['weighted avg']['recall'],
         'weighted_f1': metric['weighted avg']['f1-score']
     }
    micro_prec = precision_score(gt_labels, pred_labels, average='micro')
    micro_recall = recall_score(gt_labels, pred_labels, average='micro')
    micro_f1 = f1_score(gt_labels, pred_labels, average='micro')
    micro_metric = {
         'micro_prec': micro_prec,
         'micro_recall': micro_recall,
         'micro_f1': micro_f1
     }
    wandb.log(
         macro_metric
     )
    wandb.log(
         micro_metric
     )
    wandb.log(
         weighted_metric
     )
    wandb.log(
        {
            'Accuracy':accuracy
        }
    )
    return (result,accuracy)




def main(args, config):


    # wandb_config = wandb.config
    # print("wandb.config:", wandb_config)

    
    # config['batch_size_train'] = wandb_config.get('batch_size_train', config.get('batch_size_train'))
    # config['batch_size_test'] = wandb_config.get('batch_size_test', config.get('batch_size_test'))
    # config['optimizer']['opt'] = wandb_config.get('optimizer', {}).get('opt', config['optimizer'].get('opt'))
    # config['schedular']['sched'] = wandb_config.get('scheduler', {}).get('sched', config['schedular'].get('sched'))

    utils.init_distributed_mode(args)    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    
    
    #### Dataset #### 
    print("Creating vqa datasets")
    datasets = create_dataset('vqa', config)
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False], num_tasks, global_rank)         
    else:
        samplers = [None, None]
    
    train_loader, test_loader = create_loader(datasets,samplers,
                                            batch_size=[config['batch_size_train'],config['batch_size_test']],
                                            num_workers=[4,4],is_trains=[True, False], 
                                            collate_fns=[vqa_collate_fn,None]) 

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model #### 
    print("Creating model")
    model = PreVQA(config=config, text_encoder=args.text_encoder, text_decoder=args.text_decoder, tokenizer=tokenizer)
    model = model.to(device)   
    
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)          
    
    #load ckpt   
    if args.checkpoint and not args.evaluate:
        state_dict = torch.load(args.checkpoint, map_location='cpu')
        #checkpoint = torch.load(args.checkpoint, map_location='cpu')
        #state_dict = checkpoint['model']
        
        # reshape positional embedding to accomodate for image resolution change
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)         
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped   
        
        if not args.evaluate:
            if config['distill']:
                m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)   
                state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped 
                
            # for key in list(state_dict.keys()):
            #     if 'bert' in key:
            #         encoder_key = key.replace('bert.','')
            #         state_dict[encoder_key] = state_dict[key]
            #     # intialize text decoder as multimodal encoder (last 6 layers of model.text_encoder)
            #     if 'text_encoder' in key:
            #         if 'layer' in key:
            #             encoder_keys = key.split('.')
            #             layer_num = int(encoder_keys[4])
            #             if layer_num<6:
            #                 del state_dict[key]
            #                 continue
            #             else:
            #                 decoder_layer_num = (layer_num-6)
            #                 encoder_keys[4] = str(decoder_layer_num)
            #                 encoder_key = '.'.join(encoder_keys)
            #         else:
            #             encoder_key = key
            #         decoder_key = encoder_key.replace('text_encoder','text_decoder')
            #         state_dict[decoder_key] = state_dict[key]
            #
            #         del state_dict[key]
                
        # msg = model.load_state_dict(state_dict,strict=False)  

        # print('load checkpoint from %s'%args.checkpoint)
        # print(msg)  
        
        
        state_dict1 = torch.load(args.checkpoint_before_vqa2, map_location='cpu')['model']
        for key in list(state_dict1.keys()):
            
            if 'text_encoder' in key and 'text_encoder_m' not in key:
                
                if 'layer' in key:
                    encoder_keys = key.split('.')
                    layer_num = int(encoder_keys[4])
                    
                    if layer_num >= 6:
                        decoder_layer_num = (layer_num-6)
                        encoder_keys[4] = str(decoder_layer_num)
                        encoder_key = '.'.join(encoder_keys)
                else:
                    encoder_key = key
                if 'bert' in encoder_key:
                    encoder_key = encoder_key.replace('bert.','')
                decoder_key = encoder_key.replace('text_encoder','fusion_encoder')
                state_dict1[decoder_key] = state_dict1[key]
            del state_dict1[key]
        
        state_dict.update(state_dict1)
        msg = model.load_state_dict(state_dict,strict=False)  
        print(f'load checkpoint from {args.checkpoint} and ALBEF.fusion ##new , fuse the img and pseduo prompt after inversion')
        print(msg) 
    elif args.checkpoint and args.evaluate:
        state_dict = torch.load(args.checkpoint, map_location='cpu')
        msg = model.load_state_dict(state_dict,strict=False)  
        print(f'load checkpoint from {args.checkpoint} ,evaluate.')
        print(msg) 
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    
    
    
    print("Start training")
    start_time = time.time()

    for epoch in range(start_epoch, max_epoch):
        if epoch>0:
            lr_scheduler.step(epoch+warmup_steps)

        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config)

        if args.evaluate:
            break

        
        if utils.is_main_process():
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,
                        }
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")

            vqa_result,metric = evaluation(model, test_loader, tokenizer, device, config)  
            print(metric)   
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
    
            }
        
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))

        #dist.barrier()
    
        
    vqa_result,metric = evaluation(model, test_loader, tokenizer, device, config)      
    result_file = save_result(vqa_result, args.result_dir, 'vqa_result_epoch%d'%epoch)
                    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    
     
    
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,default='./configs/vqa.yaml') 
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--checkpoint_before_vqa2', default='') 
    parser.add_argument('--output_dir', type=str,default='output/vqa')
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--text_decoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    args = parser.parse_args()


    config = yaml.load(open(args.config, 'r', errors='ignore'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    
    
    main(args, config)
    # wandb.agent(sweep_id, function=lambda: main(args, config), count=5)
