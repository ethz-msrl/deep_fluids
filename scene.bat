REM run in script directory

REM 2d
..\manta\build_nogui\Release\manta.exe ./scene/liquid_pos_size.py --num_src_x_pos=10 --num_src_radius=4 --num_frames=200 --max_frames=199 --num_simulations=8000 --log_dir=data/liquid_pos10_size4_f200
REM ..\manta\build_nogui\Release\manta.exe ./scene/smoke_pos_size.py --num_src_x_pos=21 --num_src_radius=5 --num_frames=200 --max_frames=199 --num_simulations=21000 --log_dir=data/smoke_pos21_size5_f200
REM ..\manta\build_nogui\Release\manta.exe ./scene/smoke_pos.py --num_src_x_pos=10 --num_frames=200 --max_frames=199 --num_simulations=2000 --clamp_mode=2 --time_step=0.5 --strength=0.1 --log_dir=D:/Polybox/dev/deep-fluids/data/smoke_pos10_f100_vc
REM ..\manta\build_nogui\Release\manta.exe ./scene/smoke_pos.py --num_src_x_pos=10 --num_frames=100 --max_frames=99 --num_simulations=1000 --clamp_mode=1 --time_step=1 --log_dir=D:/Polybox/dev/deep-fluids/data/smoke_pos10_f100

REM 3d
REM ..\manta\build_nogui_vdb\Release\manta.exe ./scene/smoke3_pos.py --num_src_x_pos=10 --num_frames=100 --max_frames=99 --num_simulations=1000 --clamp_mode=1 --time_step=1 --log_dir=D:/Polybox/dev/deep-fluids/data/smoke3_pos10_f100

