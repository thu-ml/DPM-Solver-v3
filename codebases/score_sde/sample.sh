CKPT_PATH="checkpoints/cifar10_ddpmpp_deep_continuous/checkpoint_8.pth"
CONFIG="configs/vp/cifar10_ddpmpp_deep_continuous.py"
for steps in 5 6 8 10 12 15 20 25; do

if [ $steps -le 10 ]; then
    EPS="1e-3"
    STATS_DIR="statistics/cifar10_ddpmpp_deep_continuous/0.001_1200_4096"
    if [ $steps -le 8 ]; then
        if [ $steps -le 5 ]; then
            p_pseudo="True"
            lower_order_final="False"
            use_corrector="True"
        else
            p_pseudo="False"
            lower_order_final="True"
            use_corrector="True"
        fi
    else
        p_pseudo="False"
        lower_order_final="True"
        use_corrector="False"
    fi
else
    STATS_DIR="statistics/cifar10_ddpmpp_deep_continuous/0.0001_1200_4096"
    EPS="1e-4"
    p_pseudo="False"
    lower_order_final="True"
    use_corrector="True"
fi

python sample.py --config=$CONFIG --ckp_path=$CKPT_PATH --sample_folder="DPM-Solver++_"$steps --config.sampling.method=dpm_solver --config.sampling.steps=$steps --config.sampling.eps=$EPS
python sample.py --config=$CONFIG --ckp_path=$CKPT_PATH --sample_folder="UniPC_bh1_"$steps --config.sampling.method=uni_pc --config.sampling.steps=$steps --config.sampling.variant=bh1 --config.sampling.eps=$EPS
python sample.py --config=$CONFIG --ckp_path=$CKPT_PATH --sample_folder="UniPC_bh2_"$steps --config.sampling.method=uni_pc --config.sampling.steps=$steps --config.sampling.variant=bh2 --config.sampling.eps=$EPS

python sample.py --config=$CONFIG --ckp_path=$CKPT_PATH --statistics_dir=$STATS_DIR --sample_folder="DPM-Solver-v3_"$steps --config.sampling.method=dpm_solver_v3 --config.sampling.eps=$EPS --config.sampling.steps=$steps --config.sampling.predictor_pseudo=$p_pseudo --config.sampling.use_corrector=$use_corrector --config.sampling.lower_order_final=$lower_order_final --config.sampling.corrector_pseudo=False
done