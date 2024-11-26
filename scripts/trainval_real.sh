#!/bin/bash 

scenedir=$1
outputdir=$2

python train_temporal.py -s $scenedir --eval \
    	--posbase_pe=10 --timebase_pe=10 --defor_depth=6 --net_width=256 --use_skips \
        --model_path $outputdir \
        --downsample 1 --sample_interval 1 \
        --fix_until_iter 3000 --init_mode_gaussian \
        --densify_until_iter 20_000 --opacity_reset_interval 3000 \
        --iterations=40000 --defor_lr_max_steps=30000 --position_lr_max_steps=30000 --scaling_lr_max_steps=30000 --rotation_lr_max_steps=40000 \
        --white_background \
        --stop_gradient \
        --l1_l2_switch 20000 --defor_lr 0.001\
        --num_pts 100000 \
        --enable_static \
        --disable_offopa --mult_quaternion
    


python render_temporal.py  --eval \
    --posbase_pe=10 --timebase_pe=10 --defor_depth=6 --net_width=256 --use_skips \
    --model_path $outputdir \
    --downsample 1 \
    --white_background \
    --enable_static \
    --disable_offopa --mult_quaternion