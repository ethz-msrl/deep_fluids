REM liquid vis slowmo
python main.py --is_3d=True --use_curl=False --num_conv=3 --test_batch_size=3 --test_intv=150 --dataset=liquid3_mid_uni_vis4_f150 --res_x=96 --res_y=72 --res_z=48 --load_path=log/de/velocity/liquid3_mid_uni_vis4_f150/final/eval --is_train=False --test_slowmo=True

REM REM liquid drop slowmo
REM python main.py --is_3d=True --use_curl=False --num_conv=3 --test_batch_size=3 --test_intv=150 --dataset=liquid3_d5_r10_f150 --res_x=96 --res_y=48 --res_z=96 --load_path=log/de/velocity/liquid3_d5_r10_f150/final/eval --is_train=False --test_slowmo=True

REM REM liquid vis
REM python main.py --is_3d=True --use_curl=False --num_conv=3 --test_batch_size=3 --test_intv=150 --dataset=liquid3_mid_uni_vis4_f150 --res_x=96 --res_y=72 --res_z=48 --load_path=log/de/velocity/liquid3_mid_uni_vis4_f150/final/eval --is_train=False

REM REM liquid drop
REM python main.py --is_3d=True --use_curl=False --num_conv=3 --test_batch_size=3 --test_intv=150 --dataset=liquid3_d5_r10_f150 --res_x=96 --res_y=48 --res_z=96 --load_path=log/de/velocity/liquid3_d5_r10_f150/final/eval --is_train=False

REM REM upres
REM python main.py --is_3d=True --test_batch_size=2 --filters=64 --test_intv=150 --dataset=smoke3_res5_96_f150 --res_x=96 --res_y=144 --res_z=96 --load_path=log/de/velocity/smoke3_res5_96_f150/final/eval --is_train=False

REM REM smokeobs
REM python main.py --is_3d=True --test_batch_size=5 --test_intv=150 --dataset=smoke3_obs11_buo4_f150 --res_x=64 --res_y=96 --res_z=64 --load_path=log/de/velocity/smoke3_obs11_buo4_f150/final/eval --is_train=False

REM REM smokegun
REM python main.py --is_3d=True --test_batch_size=5 --test_intv=250 --dataset=smoke3_vel5_buo3_f250 --res_x=112 --res_y=64 --res_z=32 --load_path=log/de/velocity/smoke3_vel5_buo3_f250/final/eval --is_train=False

REM REM liquid 2d
REM python main.py --use_curl=False --num_conv=3 --dataset=liquid_pos10_size4_f200 --res_x=128 --res_y=64 --load_path=log/de/velocity/liquid_pos10_size4_f200/final/eval --is_train=False

REM REM smoke 2d
REM python main.py --dataset=smoke_pos21_size5_f200 --res_x=96 --res_y=128 --load_path=log/de/velocity/smoke_pos21_size5_f200/final/eval --is_train=False
