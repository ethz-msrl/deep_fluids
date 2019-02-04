python main.py --arch=de --use_curl=True
python main.py --arch=de --use_curl=False
python main.py --arch=dg --use_curl=True
python main.py --arch=dg --use_curl=False

REM python main.py --arch=dg
REM python main.py --arch=de

REM python main.py --arch=dg --is_3d=True --dataset=smoke3_vel5_buo3_f250 --res_x=112 --res_y=64 --res_z=32 --batch_size=4
REM python main.py --arch=dg --is_3d=True --dataset=smoke3_vel5_buo3_f250 --res_x=112 --res_y=64 --res_z=32 --batch_size=4 --is_train=False --load_path=log/smoke3_vel5_buo3_f250/0129_154415_dg_tag --test_batch_size=5