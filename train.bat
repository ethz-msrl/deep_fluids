

python main.py --title=final --tag=test --is_train=False --use_curl=False --w1=1 --w2=1 --max_step=300000 --dataset=liquid_pos12_size4_f200 --res_x=128 --res_y=64 --load_path=log/de/velocity/liquid_pos10_size4_f200/final/eval --test_intv=200 --test_batch_size=100 --num_conv=3
python main.py --title=final --tag=test --is_train=False --use_curl=False --w1=1 --w2=1 --max_step=300000 --dataset=liquid_pos12_size4_f200 --res_x=128 --res_y=64 --load_path=log/de/velocity/liquid_pos12_size4_f200/final/0604_191549_test --test_intv=200 --test_batch_size=100
python main.py --title=final --tag=test --is_train=False --use_curl=False --w1=1 --w2=1 --max_step=300000 --dataset=liquid_pos14_size4_f200 --res_x=128 --res_y=64 --load_path=log/de/velocity/liquid_pos14_size4_f200/final/0605_082050_test --test_intv=200 --test_batch_size=100

REM python main.py --title=final --tag=test --use_curl=False --w1=1 --w2=1 --max_step=300000 --dataset=liquid_pos12_size4_f200 --res_x=128 --res_y=64 --load_path=log/de/velocity/liquid_pos12_size4_f200/final/0604_191549_test --start_step=199976
REM python main.py --title=final --tag=test --use_curl=False --w1=1 --w2=1 --max_step=300000 --dataset=liquid_pos14_size4_f200 --res_x=128 --res_y=64

REM 3d
REM python main.py --title=lr_test --tag=2e-4_1e-6  --optimizer=adam --lr_update=test --lr_min=0.000001 --lr_max=0.0002 --log_step=10 --max_step=2000 --is_3d=True --res_x=112 --res_y=64 --res_z=32 --dataset=smoke3_vel5_buo3_f250 --batch_size=4 --num_worker=1 --test_batch_size=5

REM python main.py --dataset=ecmwf_era_interim --res_x=480 --res_y=240 --batch_size=6 --repeat=5

REM python main.py --archi=de --tag=11f5      --data_type=velocity --max_step=1000 --dataset=smoke_pos21_size5_f200 --height=128 --width=96
REM python main.py --archi=de --tag=c3        --num_conv=3 --data_type=velocity --max_step=1000 --dataset=smoke_pos21_size5_f200 --height=128 --width=96
REM python main.py --archi=de --tag=c3_64     --num_conv=3 --filters=64 --data_type=velocity --max_step=1000 --dataset=smoke_pos21_size5_f200 --height=128 --width=96
REM python main.py --archi=de --tag=uc5_10    --use_c=True --w1=1 --w2=0 --data_type=velocity --max_step=1000 --dataset=smoke_pos21_size5_f200 --height=128 --width=96
REM python main.py --archi=de --tag=uc5_11    --use_c=True --w1=1 --w2=1 --data_type=velocity --max_step=1000 --dataset=smoke_pos21_size5_f200 --height=128 --width=96
REM python main.py --archi=de --tag=uc3_10_64 --use_c=True --num_conv=3 --filters=64 --w1=1 --w2=0 --data_type=velocity --max_step=1000 --dataset=smoke_pos21_size5_f200 --height=128 --width=96
REM python main.py --archi=de --tag=gd        --optimizer=gd --data_type=velocity --max_step=1000 --dataset=smoke_pos21_size5_f200 --height=128 --width=96
REM python main.py --archi=de --tag=cy        --optimizer=gd --lr_update=cyclic --data_type=velocity --max_step=1000 --dataset=smoke_pos21_size5_f200 --height=128 --width=96

REM find optimal range (2-4 epoch)
REM python main.py --title=lr_test --tag=2e-4_6k_1e-6  --optimizer=adam --lr_update=test --lr_min=0.000001 --lr_max=0.0002 --log_step=10 --use_curl=False --num_conv=4 --batch_size=4 --repeat=5  --data_type=velocity --max_step=6000 --dataset=ecmwf_era_interim --res_x=480 --res_y=240
REM python main.py --title=skip_new --tag=test --num_conv=3
REM python main.py --title=lr_test --tag=2e-4_6k_1e-6  --optimizer=adam --lr_update=test --lr_min=0.000001 --lr_max=0.0002 --log_step=10 --use_curl=True --data_type=velocity --max_step=6000 --dataset=smoke_pos21_size5_f200 --res_y=128 --res_x=96

REM python main.py --archi=de --tag=lr_test  --optimizer=gd --lr_update=test --lr_min=0.0001  --lr_max=0.05 --data_type=velocity --max_step=12000 --dataset=smoke_pos21_size5_f200 --height=128 --width=96
REM python main.py --archi=de --tag=lr_test  --optimizer=gd --lr_update=test --lr_min=0.0005  --lr_max=0.10 --data_type=velocity --max_step=12000 --dataset=smoke_pos21_size5_f200 --height=128 --width=96
REM python main.py --archi=de --tag=lr_test  --optimizer=gd --lr_update=test --lr_min=0.00005 --lr_max=0.05 --data_type=velocity --max_step=12000 --dataset=smoke_pos21_size5_f200 --height=128 --width=96
REM python main.py --archi=de --tag=lr_test_5e-4_5e-1_6k  --optimizer=gd --lr_update=test --lr_min=0.0005  --lr_max=0.50 --data_type=velocity --max_step=6000 --dataset=smoke_pos21_size5_f200 --height=128 --width=96
REM python main.py --archi=de --tag=lr_test_5e-5_5e-1_6k  --optimizer=gd --lr_update=test --lr_min=0.00005 --lr_max=0.50 --data_type=velocity --max_step=6000 --dataset=smoke_pos21_size5_f200 --height=128 --width=96
REM python main.py --archi=de --tag=lr_test_1e-5_5e-1_6k  --optimizer=gd --lr_update=test --lr_min=0.00001 --lr_max=0.50 --data_type=velocity --max_step=6000 --dataset=smoke_pos21_size5_f200 --height=128 --width=96
REM python main.py --archi=de --tag=lr_test_5e-4_2e-1_6k  --optimizer=gd --lr_update=test --lr_min=0.0005  --lr_max=0.20 --data_type=velocity --max_step=6000 --dataset=smoke_pos21_size5_f200 --height=128 --width=96
REM python main.py --archi=de --tag=lr_test_5e-5_2e-1_6k  --optimizer=gd --lr_update=test --lr_min=0.00005 --lr_max=0.20 --data_type=velocity --max_step=6000 --dataset=smoke_pos21_size5_f200 --height=128 --width=96
REM python main.py --archi=de --tag=lr_test_1e-5_2e-1_6k  --optimizer=gd --lr_update=test --lr_min=0.00001 --lr_max=0.20 --data_type=velocity --max_step=6000 --dataset=smoke_pos21_size5_f200 --height=128 --width=96
REM python main.py --archi=de --tag=lr_test_5e-4_1e-1_6k  --optimizer=gd --lr_update=test --lr_min=0.0005  --lr_max=0.10 --data_type=velocity --max_step=6000 --dataset=smoke_pos21_size5_f200 --height=128 --width=96
REM python main.py --archi=de --tag=lr_test_5e-5_1e-1_6k  --optimizer=gd --lr_update=test --lr_min=0.00005 --lr_max=0.10 --data_type=velocity --max_step=6000 --dataset=smoke_pos21_size5_f200 --height=128 --width=96
REM python main.py --archi=de --tag=lr_test_1e-5_1e-1_6k  --optimizer=gd --lr_update=test --lr_min=0.00001 --lr_max=0.10 --data_type=velocity --max_step=6000 --dataset=smoke_pos21_size5_f200 --height=128 --width=96

REM gan
REM python main.py --archi=dg --tag=11_0.001 --w1=1 --w2=1 --w_adv=0.001 --data_type=velocity --g_lr=0.00004 --dataset=smoke_pos10_f100 --height=128 --width=96

REM cnn

REM learning rate test
REM python main.py --archi=de --tag=lr_1e-4_200k --data_type=velocity --g_lr=0.00010 --lr_update_step=80000 --max_step=200000 --dataset=smoke_pos10_f100 --height=128 --width=96 --load_path=log/de/velocity/smoke_pos10_f100_0314_082913_lr_1e-4_200k --start_step=50227

REM python main.py --archi=de --tag=lr_1e-3 --data_type=velocity --g_lr=0.00100 --lr_update_step=120000 --dataset=smoke_pos10_f100 --height=128 --width=96
REM -> Nan, failed!

REM python main.py --archi=de --tag=lr_1e-4 --data_type=velocity --g_lr=0.00010 --lr_update_step=120000 --dataset=smoke_pos10_f100 --height=128 --width=96 --load_path=log/de/velocity/smoke_pos10_f100_0312_153857_lr_1e-4 --start_step=150312
REM python main.py --archi=de --tag=lr_1e-4 --data_type=velocity --g_lr=0.00010 --lr_update_step=500000 --dataset=smoke_pos10_f100 --height=128 --width=96 --load_path=log/de/velocity/smoke_pos10_f100_0312_153857_lr_1e-4_no_lr_update --start_step=226522
REM python main.py --archi=de --tag=lr_4e-5 --data_type=velocity --g_lr=0.00004 --lr_update_step=120000 --dataset=smoke_pos10_f100 --height=128 --width=96 --load_path=log/de/velocity/smoke_pos10_f100_0313_041436_11 --start_step=193895

REM python main.py --archi=de --tag=lr_1e-4 --data_type=velocity --g_lr=0.00010 --dataset=smoke_pos10_f100 --height=128 --width=96
REM python main.py --archi=de --tag=lr_4e-5 --data_type=velocity --g_lr=0.00004 --dataset=smoke_pos10_f100 --height=128 --width=96