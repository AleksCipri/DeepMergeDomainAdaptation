
#!/bin/bash

# an example of training and evaluation of the domain adaptation model. 

python train_feature.py --gpu_id <gpu_id> \
                        --net ResNet50 \
                        --dset office \
                        --test_interval 500 \
                        --snapshot_interval 10000 \
                        --ly_type cosine \
                        --loss_type mmd \
                        --fisher_loss_type tr \
                        --output_dir <output_path> \
                        --s_dset_path ../data/office31/amazon_31_list.txt \
                        --t_dset_path ../data/office31/dslr_31_list.txt \
                        --em_loss_coef 0.1 \
                        --inter_loss_coef 1. \
                        --intra_loss_coef 1. \
                        --trade_off 1.

python eval_da.py --gpu_id <gpu_id> \
                  --net ResNet50 \
                  --dset office \
                  --ly_type cosine \
                  --ckpt_path <path of the saved model> \
                  --s_dset_path ../data/office31/amazon_31_list.txt \
                  --t_dset_path ../data/office31/dslr_31_list.txt
