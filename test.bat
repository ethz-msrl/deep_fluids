REM liquid 2d
python main.py --use_curl=False --num_conv=3 --dataset=liquid_pos10_size4_f200 --res_x=128 --res_y=64 --load_path=log/de/velocity/liquid_pos10_size4_f200/final/test --is_train=False

REM smoke 2d
REM python main.py --dataset=smoke_pos21_size5_f200 --res_x=96 --res_y=128 --load_path=log/de/velocity/smoke_pos21_size5_f200/final/test --is_train=False
