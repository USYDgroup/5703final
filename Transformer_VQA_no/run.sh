# train
python   run_vqa2.py \
--config ./configs/vqa.yaml \
--output_dir output_vqa2/vqa \
--checkpoint "/home/admin1/5703-upload/5703/Transformer_VQA_no/output_vqa2/vqa/checkpoint_07.pth" \
--checkpoint_before_vqa2 "/home/admin1/5703-upload/5703/Transformer_VQA_no/ALBEF.pth" \
--evaluate

# python   run_vqa2.py \
# --config ./configs/vqa.yaml \
# --output_dir output_vqa2/vqa \
# --checkpoint "/home/admin1/5703-upload/5703/Transformer_VQA_no/ALBEF.pth" \
# --checkpoint_before_vqa2 "/home/admin1/5703-upload/5703/Transformer_VQA_no/ALBEF.pth"

#test
#python   run.py \
#--config ./configs/vqa.yaml \
#--output_dir output/vqa \
#--checkpoint "/home/admin1/5703-upload/5703/Transformer_VQA_no/checkpoint_07.pth" \
#--evaluate