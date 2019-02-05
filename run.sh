### smoke_pos_size, 2D
# 1. generate dataset
# ../manta/build/manta ./scene/smoke_pos_size.py

# 2. train
# python main.py --is_3d=False --dataset=smoke_pos21_size5_f200 --resolution_x=96 --resolution_y=128

# 3. test
# python main.py --is_train=False --load_path=MODEL_DIR --is_3d=False --dataset=smoke_pos21_size5_f200 --resolution_x=96 --resolution_y=128


### smoke3_buo_vel, 3D
# ../manta/build/manta ./scene/smoke3_buo_vel.py
# python main.py --is_3d=True --dataset=smoke3_vel5_buo3_f250 --resolution_x=112 --resolution_y=64 --resolution_z=32


### smoke3_obs_buo, 3D
# ../manta/build/manta ./scene/smoke3_obs_buo.py
# python main.py --is_3d=True --dataset=smoke3_obs11_buo4_f150 --resolution_x=64 --resolution_y=96 --resolution_z=64


### liquid_pos_size, 2D
# ../manta/build/manta ./scene/liquid_pos_size.py
# python main.py --use_curl=False --is_3d=False --dataset=liquid_pos10_size4_f200 --resolution_x=128 --resolution_y=64


### liquid3_d_r, 3D
# ../manta/build/manta ./scene/liquid3_d_r.py
# python main.py --use_curl=False --is_3d=True --dataset=liquid3_d5_r10_f150 --resolution_x=96 --resolution_y=64 --resolution_z=96


### smoke3_rot, 2D
# ../manta/build/manta ./scene/smoke3_rot.py --log_dir=data/smoke_rot_f500 --resolution_x=96 --resolution_y=128 --resolution_z=1 --open_bound=xXyY --num_dof=1

### smoke3_rot, 3D
# ../manta/build/manta ./scene/smoke3_rot.py --log_dir=data/smoke3_rot_f500 --resolution_x=48 --resolution_y=72 --resolution_z=48

### smoke3_mov, 2D
# ../manta/build/manta ./scene/smoke3_mov.py --log_dir=data/smoke_mov200_f400 --resolution_x=96 --resolution_y=128 --resolution_z=1 --open_bound=xXyY --num_dof=1

### smoke3_mov, 3D
# ../manta/build/manta ./scene/smoke3_mov.py --log_dir=data/smoke3_mov200_f400 --resolution_x=48 --resolution_y=72 --resolution_z=48