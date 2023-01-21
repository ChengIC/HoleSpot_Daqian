### for training on cityscapes
#python train.py --height 416 --width 128  --dataset cityscapes_preprocessed --split cityscapes_preprocessed --scheduler_step_size 14  --batch 16  --model_name mono_model --png --data_path data_path/cityscapes_preprocessed

### for training on kitti
log_name=log1
weights_name=/mnt/nas/kaichen/eng/DEV/DEV/DIFFNET/mono_model/log5/models/weights_20
python train.py --cuda_device 1 --scales 0 --scheduler_step_size 4 --load_weights_folder $weights_name --batch 6  --dataset kitti --model_name mono_model --png --data_path /mnt/nas/kaichen/compete/png --log_name $log_name