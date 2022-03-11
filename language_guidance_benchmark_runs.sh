#-------------------
#   CUB200
#-------------------
### ResNet50, dim = 128
## Multisimilarity + PLG
# Example schedule: --tau 28 --gamma 0.25
python main.py --seed 2 --log_online --group cub200-rn128_msim_plg --no_train_metrics --project LanguageGuidance --gpu $gpu --source_path $datapath \
--dataset cub200 --n_epochs 200 --tau 200 --gamma 1 --arch resnet50_frozen_normalize --embed_dim 128 --loss multisimilarity --bs 90 \
--language_pseudoclass --language_distill_w 7.5
## Multisimilarity + S2SD + PLG
# Example Schedule: --tau 144 --gamma 0.5
python main.py --seed 2 --log_online --group cub200-rn128_s2sd-msim_plg --project LanguageGuidance --gpu $gpu --source_path $datapath --no_train_metrics \
--dataset cub200 --n_epochs 300 --tau 300 --gamma 1 --arch resnet50_frozen_normalize --embed_dim 128 --bs 108 \
--loss s2sd --loss_s2sd_source multisimilarity --loss_s2sd_target multisimilarity --loss_s2sd_feat_distill --loss_s2sd_feat_w 125 --loss_s2sd_feat_distill_delay 0 --loss_s2sd_pool_aggr \
--language_distill_w 1.5 --language_pseudoclass
### ResNet50, dim = 512
## Multisimilarity + PLG
# Example Schedule: --tau 34 --gamma 0.25
python main.py --seed 0 --log_online --group cub200-rn512_msim_plg --no_train_metrics --project LanguageGuidance --gpu $gpu --source_path $datapath \
--dataset cub200 --n_epochs 150 --tau 150 --gamma 1 --arch resnet50_frozen_normalize --embed_dim 512 --loss multisimilarity --bs 90 \
--language_pseudoclass --language_distill_w 10
## Multisimilarity + S2SD + PLG
# Example Schedule: --tau 29 --gamma 0.5
python main.py --seed 0 --log_online --group cub200-rn512_s2sd-msim_plg --project LanguageGuidance --gpu $gpu --source_path $datapath --no_train_metrics \
--dataset cub200 --n_epochs 250 --tau 250 --gamma 1 --arch resnet50_frozen_normalize --embed_dim 512 --bs 108 \
--loss s2sd --loss_s2sd_source multisimilarity --loss_s2sd_target multisimilarity --loss_s2sd_feat_distill --loss_s2sd_feat_w 60 --loss_s2sd_feat_distill_delay 0 --loss_s2sd_pool_aggr \
--language_distill_w 2 --language_pseudoclass

#-------------------
#   CARS196
#-------------------
### ResNet50, dim = 128
## Multisimilarity + PLG
# Example Schedule: --tau 102 --gamma 0.25
python main.py --seed 1 --log_online --group cars196-rn128_msim_plg --project LanguageGuidance --gpu $gpu --source_path $datapath --dataset cars196 \
--n_epochs 250 --tau 250 --gamma 1 --arch resnet50_frozen_normalize --embed_dim 128 --bs 108 --loss multisimilarity \
--language_distill_w 6 --language_pseudoclass
## Multisimilarity + S2SD + PLG
# Example Schedule: --tau 110 --gamma 0.25
python main.py --seed 0 --log_online --group cars196-rn128_s2sd-msim_plg --project LanguageGuidance --gpu $gpu --source_path $datapath --no_train_metrics \
--dataset cars196 --n_epochs 250 --tau 250 --gamma 1 --arch resnet50_frozen_normalize --embed_dim 128 --bs 108 \
--loss s2sd --loss_s2sd_source multisimilarity --loss_s2sd_target multisimilarity --loss_s2sd_feat_distill --loss_s2sd_feat_w 50 --loss_s2sd_feat_distill_delay 0 --loss_s2sd_pool_aggr \
--language_distill_w 2.5 --language_pseudoclass
### ResNet50, dim = 512
## Multisimilarity + PLG
# Example Schedule: --tau 107 --gamma 0.25
python main.py --seed 0 --log_online --group cars196-rn512_msim_plg --project LanguageGuidance --gpu $gpu --source_path $datapath --dataset cars196 \
--n_epochs 250 --tau 250 --gamma 1 --arch resnet50_frozen_normalize --embed_dim 512 --bs 108 --loss multisimilarity \
--language_distill_w 5.5 --language_pseudoclass
## Multisimilarity + S2SD + PLG
# Example Schedule: --tau 165 --gamma 0.1
python main.py --seed 0 --log_online --group cars196-rn512_s2sd-msim_plg --project LanguageGuidance --gpu $gpu --source_path $datapath --no_train_metrics \
--dataset cars196 --n_epochs 300 --tau 300 --gamma 1 --arch resnet50_frozen_normalize --embed_dim 512 --bs 108 \
--loss s2sd --loss_s2sd_source multisimilarity --loss_s2sd_target multisimilarity --loss_s2sd_feat_distill --loss_s2sd_feat_w 40 --loss_s2sd_feat_distill_delay 0 --loss_s2sd_pool_aggr \
--language_distill_w 2 --language_pseudoclass
