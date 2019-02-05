REM ----- smoke_pos_size, 2D
REM 1. generate dataset
REM ..\manta\build\Release\manta .\scene\smoke_pos_size.py

REM 2. train
REM python main.py --is_3d=False --dataset=smoke_pos21_size5_f200 --res_x=96 --res_y=128

REM 3. test
REM python main.py --is_train=False --load_path=MODEL_DIR --test_batch_size=100 --is_3d=False --dataset=smoke_pos21_size5_f200 --res_x=96 --res_y=128


REM ----- smoke3_vel_buo, 3D
REM ..\manta\build\Release\manta .\scene\smoke3_vel_buo.py

python main.py --is_3d=True --dataset=smoke3_vel5_buo3_f250 --res_x=112 --res_y=64 --res_z=32 --batch_size=4 --num_worker=1

REM python main.py --is_train=False --load_path=MODEL_DIR --test_batch_size=5 --is_3d=True --dataset=smoke3_vel5_buo3_f250 --res_x=112 --res_y=64 --res_z=32 --batch_size=4 --num_worker=1


REM ----- smoke3_obs_buo, 3D
REM ..\manta\build\Release\manta .\scene\smoke3_obs_buo.py
REM python main.py --is_3d=True --dataset=smoke3_obs11_buo4_f150 --res_x=64 --res_y=96 --res_z=64


REM ----- liquid_pos_size, 2D
REM ..\manta\build\Release\manta .\scene\liquid_pos_size.py
REM python main.py --use_curl=False --is_3d=False --dataset=liquid_pos10_size4_f200 --res_x=128 --res_y=64


REM ----- liquid3_d_r, 3D
REM ..\manta\build\Release\manta .\scene\liquid3_d_r.py
REM python main.py --use_curl=False --is_3d=True --dataset=liquid3_d5_r10_f150 --res_x=96 --res_y=64 --res_z=96


REM ----- smoke3_rot, 2D
REM ..\manta\build\Release\manta .\scene\smoke3_rot.py --log_dir=data\smoke_rot_f500 --resolution_x=96 --resolution_y=128 --resolution_z=1 --open_bound=xXyY --num_dof=1


REM ----- smoke3_rot, 3D
REM ..\manta\build\Release\manta .\scene\smoke3_rot.py --log_dir=data\smoke3_rot_f500 --resolution_x=48 --resolution_y=72 --resolution_z=48


REM ----- smoke3_mov, 2D
REM ..\manta\build\Release\manta .\scene\smoke3_mov.py --log_dir=data\smoke_mov200_f400 --resolution_x=96 --resolution_y=128 --resolution_z=1 --open_bound=xXyY --num_dof=1


REM ----- smoke3_mov, 3D
REM ..\manta\build\Release\manta .\scene\smoke3_mov.py --log_dir=data\smoke3_mov200_f400 --resolution_x=48 --resolution_y=72 --resolution_z=48