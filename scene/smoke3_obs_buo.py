import argparse
import os
from tqdm import trange

import numpy as np
import matplotlib.pyplot as plt

import gc
try:
	from manta import *
except ImportError:
	pass

def str2bool(v):
    return v.lower() in ('true', '1')

parser = argparse.ArgumentParser()

parser.add_argument("--log_dir", type=str, default='data/smoke3_obs11_buo4_f150')

parser.add_argument("--num_param", type=int, default=3)
parser.add_argument("--path_format", type=str, default='%d_%d_%d.npz')
parser.add_argument("--p0", type=str, default='obs_x_pos')
parser.add_argument("--p1", type=str, default='buoyancy')
parser.add_argument("--p2", type=str, default='frames')

parser.add_argument("--min_obs_x_pos", type=float, default=0.2)
parser.add_argument("--max_obs_x_pos", type=float, default=0.8)
parser.add_argument("--num_obs_x_pos", type=int, default=11)
parser.add_argument("--obs_radius", type=float, default=0.15)
parser.add_argument("--obs_y_pos", type=float, default=0.5)
parser.add_argument("--obs_z_pos", type=float, default=0.5)
parser.add_argument("--min_buoyancy", type=float, default=-8e-3)
parser.add_argument("--max_buoyancy", type=float, default=-16e-3)
parser.add_argument("--num_buoyancy", type=int, default=4)
parser.add_argument("--src_x_pos", type=float, default=0.5)
parser.add_argument("--src_y_pos", type=float, default=0.13)
parser.add_argument("--src_z_pos", type=float, default=0.5)
parser.add_argument("--src_radius", type=float, default=0.12)
parser.add_argument("--min_frames", type=int, default=0)
parser.add_argument("--max_frames", type=int, default=149)
parser.add_argument("--num_frames", type=int, default=150)
parser.add_argument("--num_simulations", type=int, default=2500)

parser.add_argument("--resolution_x", type=int, default=64)
parser.add_argument("--resolution_y", type=int, default=96)
parser.add_argument("--resolution_z", type=int, default=64)
parser.add_argument("--bWidth", type=int, default=1)
parser.add_argument("--open_bound", type=bool, default=False)
parser.add_argument("--time_step", type=float, default=0.5)
parser.add_argument("--clamp_mode", type=int, default=2)

parser.add_argument('--is_test', type=str2bool, default=False)
parser.add_argument('--vpath', type=str, default='')

args = parser.parse_args()


def get_param(p1, p2):
    min_p1 = args.min_obs_x_pos
    max_p1 = args.max_obs_x_pos
    num_p1 = args.num_obs_x_pos
    min_p2 = args.min_buoyancy
    max_p2 = args.max_buoyancy
    num_p2 = args.num_buoyancy
    p1_ = p1/(num_p1-1) * (max_p1-min_p1) + min_p1
    p2_ = p2/(num_p2-1) * (max_p2-min_p2) + min_p2
    return p1_, p2_

def test():
    filename = os.path.basename(args.vpath)[:-4]
    print(filename)

    p1, p2 = filename.split('_')
    p1, p2 = get_param(float(p1), float(p2))
    title = 'd_' + filename
    vdb_dir = os.path.join(os.path.dirname(args.vpath), filename)

    # solver params
    res_x = args.resolution_x
    res_y = args.resolution_y
    res_z = args.resolution_z
    gs = vec3(res_x, res_y, res_z)

    s = Solver(name='main', gridSize=gs, dim=3)
    s.timestep = args.time_step

    flags = s.create(FlagGrid)
    vel = s.create(MACGrid)
    density = s.create(RealGrid)

    flags.initDomain(boundaryWidth=args.bWidth)
    flags.fillGrid()
        
    density.clear()
    vel.clear()

    src_radius = gs.x*args.src_radius
    src_center = gs*vec3(args.src_x_pos,args.src_y_pos,args.src_z_pos)
    source = s.create(Sphere, center=src_center, radius=src_radius)

    obs_radius = gs.x*args.obs_radius
    obs_center = gs*vec3(p1,args.obs_y_pos,args.obs_z_pos)
    obs = s.create(Sphere, center=obs_center, radius=obs_radius)
    obs.applyToGrid(grid=flags, value=FlagObstacle)

    with np.load(args.vpath) as data:
        x = data['v']

    for i in range(args.num_frames):
        v = x[i,...]
        copyArrayToGridMAC(target=vel, source=v)

        source.applyToGrid(grid=density, value=1)
        advectSemiLagrange(flags=flags, vel=vel, grid=density, order=2, 
                           openBounds=args.open_bound, boundaryWidth=args.bWidth, clampMode=args.clamp_mode)

        vdb_file_path = os.path.join(vdb_dir, title+'_%d.vdb' % i)
        density.save(vdb_file_path)
        s.step()

def main():
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    field_type = ['v', 'vdb'] # 's', 'p']
    for field in field_type:
        field_path = os.path.join(args.log_dir,field)
        if not os.path.exists(field_path):
            os.mkdir(field_path)

    args_file = os.path.join(args.log_dir, 'args.txt')
    with open(args_file, 'w') as f:
        for k, v in vars(args).items():
            print('  %s: %s' % (k, v))
            f.write('%s: %s\n' % (k, v))

    p1_space = np.linspace(args.min_obs_x_pos, 
                            args.max_obs_x_pos,
                            args.num_obs_x_pos)
    p2_space = np.linspace(args.min_buoyancy,
                            args.max_buoyancy,
                            args.num_buoyancy)
    p_list = np.array(np.meshgrid(p1_space, p2_space)).T.reshape(-1, 2)
    pi1_space = range(args.num_obs_x_pos)
    pi2_space = range(args.num_buoyancy)
    pi_list = np.array(np.meshgrid(pi1_space, pi2_space)).T.reshape(-1, 2)

    # solver params
    res_x = args.resolution_x
    res_y = args.resolution_y
    res_z = args.resolution_z
    gs = vec3(res_x, res_y, res_z)

    s = Solver(name='main', gridSize=gs, dim=3)
    s.timestep = args.time_step

    flags = s.create(FlagGrid)
    vel = s.create(MACGrid)
    density = s.create(RealGrid)
    pressure = s.create(RealGrid)

    # omega = s.create(VecGrid)
    # stream = s.create(VecGrid)
    # vel_out = s.create(MACGrid)

    v_ = np.zeros([res_y,res_x,res_z,3], dtype=np.float32)
    # d_ = np.zeros([res_z,res_y,res_x], dtype=np.float32)
    # s_ = np.zeros([res_y,res_x,res_z,3], dtype=np.float32)
    # p_ = np.zeros([res_z,res_y,res_x], dtype=np.float32)
    
    if (GUI):
        gui = Gui()
        gui.show(True)
        gui.nextVec3Display()
        gui.nextVec3Display()
        gui.nextVec3Display()
        #gui.pause()

    print('start generation')
    v_range = [np.finfo(np.float).max, np.finfo(np.float).min]
    # d_range = [np.finfo(np.float).max, np.finfo(np.float).min]
    # s_range = [np.finfo(np.float).max, np.finfo(np.float).min]
    # p_range = [np.finfo(np.float).max, np.finfo(np.float).min]
    for i in trange(len(p_list), desc='scenes'):
        flags.initDomain(boundaryWidth=args.bWidth)
        flags.fillGrid()
        # if args.open_bound:
        # 	setOpenBound(flags, args.bWidth,'xXyY', FlagOutflow|FlagEmpty)

        density.clear()
        vel.clear()
        pressure.clear()
        # stream.clear()
        
        src_radius = gs.x*args.src_radius
        src_center = gs*vec3(args.src_x_pos,args.src_y_pos,args.src_z_pos)
        source = s.create(Sphere, center=src_center, radius=src_radius)

        p0, p1 = p_list[i][0], p_list[i][1]
        obs_radius = gs.x*args.obs_radius
        obs_center = gs*vec3(p0,args.obs_y_pos,args.obs_z_pos)
        obs = s.create(Sphere, center=obs_center, radius=obs_radius)
        obs.applyToGrid(grid=flags, value=FlagObstacle)
        
        buoyancy = vec3(0,p1,0)		
        for t in trange(args.num_frames, desc='sim'):
            source.applyToGrid(grid=density, value=1)
                
            advectSemiLagrange(flags=flags, vel=vel, grid=density, order=2,
                                openBounds=args.open_bound, boundaryWidth=args.bWidth, clampMode=args.clamp_mode)
            advectSemiLagrange(flags=flags, vel=vel, grid=vel,     order=2,
                                openBounds=args.open_bound, boundaryWidth=args.bWidth, clampMode=args.clamp_mode)
            setWallBcs(flags=flags, vel=vel)
            addBuoyancy(density=density, vel=vel, gravity=buoyancy, flags=flags)
            solvePressure(flags=flags, vel=vel, pressure=pressure, cgMaxIterFac=10.0, cgAccuracy=0.0001)
            setWallBcs(flags=flags, vel=vel)
            
            # # get streamfunction
            # curl1(vel, omega)
            # solve_stream_function_pcg(flags, omega, stream)
            # curl2(stream, vel)

            copyGridToArrayMAC(target=v_, _Source=vel)
            # copyGridToArrayReal(target=d_, source=density)
            # copyGridToArrayReal(target=p_, source=pressure)
            # copyGridToArrayVec3(target=s_, source=stream)

            v_range = [np.minimum(v_range[0], v_.min()),
                        np.maximum(v_range[1], v_.max())]
            # d_range = [np.minimum(d_range[0], d_.min()),
            # 		   np.maximum(d_range[1], d_.max())]
            # s_range = [np.minimum(s_range[0], s_.min()),
            # 		   np.maximum(s_range[1], s_.max())]
            # p_range = [np.minimum(p_range[0], p_.min()),
            # 		   np.maximum(p_range[1], p_.max())]

            pit = tuple(pi_list[i].tolist() + [t])
            param_ = p_list[i].tolist() + [t]
            
            v_file_path = os.path.join(args.log_dir, 'v', args.path_format % pit)
            np.savez_compressed(v_file_path, 
                                x=v_, # yxzd 
                                y=param_)

            if 'vdb' in field_type:
                vdb_file_path = os.path.join(args.log_dir, 'vdb', '%d_%d_%d.vdb' % pit)
                density.save(vdb_file_path)

            # d_file_path = os.path.join(args.log_dir, 'd', args.path_format % pit)
            # np.savez_compressed(d_file_path, 
            # 					x=np.expand_dims(d_, axis=-1),
            # 					y=param_)

            # s_file_path = os.path.join(args.log_dir, 's', args.path_format % pit)
            # np.savez_compressed(s_file_path, 
            # 					x=s_, # yxzd
            # 					y=param_)

            # p_file_path = os.path.join(args.log_dir, 'p', args.path_format % pit)
            # np.savez_compressed(p_file_path, 
            # 					x=np.expand_dims(p_, axis=-1),
            # 					y=param_)

            s.step()
        gc.collect()

    vrange_file = os.path.join(args.log_dir, 'v_range.txt')
    with open(vrange_file, 'w') as f:
        print('velocity min %.3f max %.3f' % (v_range[0], v_range[1]))
        f.write('%.3f\n' % v_range[0])
        f.write('%.3f' % v_range[1])

    # drange_file = os.path.join(args.log_dir, 'd_range.txt')
    # with open(drange_file, 'w') as f:
    # 	print('%s: density min %.3f max %.3f' % (datetime.now(), d_range[0], d_range[1]))
    # 	f.write('%.3f\n' % d_range[0])
    # 	f.write('%.3f' % d_range[1])

    # srange_file = os.path.join(args.log_dir, 's_range.txt')
    # with open(srange_file, 'w') as f:
    # 	print('%s: stream min %.3f max %.3f' % (datetime.now(), s_range[0], s_range[1]))
    # 	f.write('%.3f\n' % s_range[0])
    # 	f.write('%.3f' % s_range[1])

    # prange_file = os.path.join(args.log_dir, 'p_range.txt')
    # with open(prange_file, 'w') as f:
    # 	print('%s: pressure min %.3f max %.3f' % (datetime.now(), p_range[0], p_range[1]))
    # 	f.write('%.3f\n' % p_range[0])
    # 	f.write('%.3f' % p_range[1])

    print('Done')


if __name__ == '__main__':
    if args.is_test:
        test()
    else:
        main()