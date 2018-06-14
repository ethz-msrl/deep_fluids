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

def blend_test():
    # os.chdir(os.path.dirname(__file__) + '/..')
    # print(os.getcwd())

    v_gt_path = os.path.join(args.log_dir, 'smoke3_obs1_buo1_intp_f150', 'v', '0_0_149.npz')
    with np.load(v_gt_path) as data:
        v_gt = data['x']
        v_gt = v_gt[10:96-10,10:64-10,10:64-10,:]
    nz_ = np.nonzero(v_gt)
    z = np.ones(v_gt.shape)
    z[nz_] = 0    

    def jacobian_np3(x):
        # x: bzyxd
        dudx = x[:,:,:,1:,0] - x[:,:,:,:-1,0]
        dvdx = x[:,:,:,1:,1] - x[:,:,:,:-1,1]
        dwdx = x[:,:,:,1:,2] - x[:,:,:,:-1,2]
        dudy = x[:,:,:-1,:,0] - x[:,:,1:,:,0] # y fliped
        dvdy = x[:,:,:-1,:,1] - x[:,:,1:,:,1] # y fliped
        dwdy = x[:,:,:-1,:,2] - x[:,:,1:,:,2] # y fliped
        dudz = x[:,1:,:,:,0] - x[:,:-1,:,:,0]
        dvdz = x[:,1:,:,:,1] - x[:,:-1,:,:,1]
        dwdz = x[:,1:,:,:,2] - x[:,:-1,:,:,2]

        dudx = np.concatenate((dudx, np.expand_dims(dudx[:,:,:,-1], axis=3)), axis=3)
        dvdx = np.concatenate((dvdx, np.expand_dims(dvdx[:,:,:,-1], axis=3)), axis=3)
        dwdx = np.concatenate((dwdx, np.expand_dims(dwdx[:,:,:,-1], axis=3)), axis=3)

        dudy = np.concatenate((np.expand_dims(dudy[:,:,0,:], axis=2), dudy), axis=2)
        dvdy = np.concatenate((np.expand_dims(dvdy[:,:,0,:], axis=2), dvdy), axis=2)
        dwdy = np.concatenate((np.expand_dims(dwdy[:,:,0,:], axis=2), dwdy), axis=2)

        dudz = np.concatenate((dudz, np.expand_dims(dudz[:,-1,:,:], axis=1)), axis=1)
        dvdz = np.concatenate((dvdz, np.expand_dims(dvdz[:,-1,:,:], axis=1)), axis=1)
        dwdz = np.concatenate((dwdz, np.expand_dims(dwdz[:,-1,:,:], axis=1)), axis=1)

        u = dwdy - dvdz
        v = dudz - dwdx
        w = dvdx - dudy
        
        j = np.stack([dudx,dudy,dudz,dvdx,dvdy,dvdz,dwdx,dwdy,dwdz], axis=-1)
        c = np.stack([u,v,w], axis=-1)
        
        return j, c    

    def saveimg(x, filename):
        x /= x.max()
        x = np.uint8(plt.cm.afmhot(x)*255)
        from PIL import Image
        im = Image.fromarray(x)
        im.save(filename)

    from scipy import interpolate
    for kind in ['linear']: # , 'nearest', 'zero', 'slinear']:
        
        mae_list = []
        mse_list = []
        mae_gt_list = []
        mse_gt_list = []
        vnum = np.count_nonzero(z)
        for j in range(args.num_frames):
            v_gt_path = os.path.join(args.log_dir, 'v', '4_2_%d.npz' % j)
            with np.load(v_gt_path) as data:
                v1 = data['x']
            
            v_gt_path = os.path.join(args.log_dir, 'v', '5_2_%d.npz' % j)
            with np.load(v_gt_path) as data:
                v2 = data['x']

            v12 = np.stack((v1, v2), axis=0)
            print(v12.shape)

            f_intp = interpolate.interp1d(range(2), v12, axis=0, kind=kind)
            v = f_intp(0.5)
            
            # if j == 80:
            #     v = v.transpose([2,0,1,3])
            #     v = v[np.newaxis,:,::-1,:,:]
                    
            #     _, c = jacobian_np3(v)
            #     c = np.sqrt(np.sum(c**2, axis=-1))[0]
            #     v = np.sqrt(np.sum(v**2, axis=-1))[0]
            #     idx_mid = int(args.resolution_z*0.5)
            #     # plt.subplot(121)
            #     # plt.imshow(v[idx_mid,:,:], cmap=plt.cm.afmhot)
            #     # plt.subplot(122)
            #     # plt.imshow(c[idx_mid,:,:], cmap=plt.cm.afmhot)
            #     # plt.show()        

            #     vort_mag_path = os.path.join('vort_mag_%s.png' % kind)
            #     saveimg(c[idx_mid,:,:], vort_mag_path)
            #     vel_mag_path = os.path.join('vel_mag_%s.png' % kind)
            #     saveimg(v[idx_mid,:,:], vel_mag_path)
            #     exit()
                
            v = v[10:96-10,10:64-10,10:64-10,:]
            v_ = np.multiply(v, z)
            # plt.subplot(121)
            # plt.imshow(v_[40,:,:,0])
            # plt.subplot(122)
            # plt.imshow(v[40,:,:,0])
            # plt.show()
            mae_err = np.sum(np.abs(v_))/vnum # mean absolute error
            mse_err = np.sum(v_**2)/vnum # mean absolute error
            mae_list.append(mae_err)
            mse_list.append(mse_err)

        save_dir = 'log/de/velocity/smoke3_obs11_buo4_f150/final/eval/p2_n150'
        mae_path = os.path.join(save_dir, '%s_2_linear_mae.csv' % str(p[i]))
        np.savetxt(mae_path, mae_list, delimiter=',')                
        

def obs_penetration_test():
    os.chdir(os.path.dirname(__file__) + '/..')

    nz = []
    v_gt_path = os.path.join(args.log_dir, 'v', '4_2_149.npz')
    with np.load(v_gt_path) as data:
        v_gt = data['x']
        v_gt = v_gt[10:96-10,10:64-10,10:64-10,:]
    nz_ = np.nonzero(v_gt)
    z = np.ones(v_gt.shape)
    z[nz_] = 0    
    nz.append(z)

    v_gt_path = os.path.join(args.log_dir, 'smoke3_obs1_buo1_intp_f150', 'v', '0_0_149.npz')
    with np.load(v_gt_path) as data:
        v_gt = data['x']
        v_gt = v_gt[10:96-10,10:64-10,10:64-10,:]
    nz_ = np.nonzero(v_gt)
    z = np.ones(v_gt.shape)
    z[nz_] = 0    
    nz.append(z)

    v_gt_path = os.path.join(args.log_dir, 'v', '5_2_149.npz')
    with np.load(v_gt_path) as data:
        v_gt = data['x']
        v_gt = v_gt[10:96-10,10:64-10,10:64-10,:]
    nz_ = np.nonzero(v_gt)
    z = np.ones(v_gt.shape)
    z[nz_] = 0    
    nz.append(z)

    # print(os.getcwd())
    p = [4, 4.5, 5]
    for i, p0 in enumerate([0.44, 0.47, 0.5]):
        v_path = os.path.join('log/de/velocity/smoke3_obs11_buo4_f150/final/eval/p2_n150', '%s_2.npz' % str(p[i]))
        with np.load(v_path) as data:
            x = data['v']
            
        mae_list = []
        mse_list = []
        mae_gt_list = []
        mse_gt_list = []
        vnum = np.count_nonzero(nz[i])
        for j in range(args.num_frames):
            v = x[j,...]
            v = v[10:96-10,10:64-10,10:64-10,:]
            v_ = np.multiply(v, nz[i])
            # plt.subplot(121)
            # plt.imshow(v_[40,:,:,0])
            # plt.subplot(122)
            # plt.imshow(v[40,:,:,0])
            # plt.show()
            mae_err = np.sum(np.abs(v_))/vnum # mean absolute error
            mse_err = np.sum(v_**2)/vnum # mean absolute error
            mae_list.append(mae_err)
            mse_list.append(mse_err)

        # obs_path = os.path.join(args.log_dir, 'obs', '%f.npz' % p0)
        # with np.load(obs_path) as data:
        #     obs_flag = data['x']
        #     vnum = np.count_nonzero(obs_flag)
            # # debug
            # obs_flag = obs_flag.transpose([2,0,1,3])
            # idx_mid = int(args.resolution_z*0.5)
            # plt.imshow(obs_flag[idx_mid,:,:,0])
            # plt.show()
        
        # obs_path = os.path.join(args.log_dir, 'obs', '%f_big.npz' % p0)
        # with np.load(obs_path) as data:
        #     obs_big_flag = data['x']

        # v_gt_path = os.path.join(args.log_dir, 'v', '%s_2_%d.npz' % (str(p[i]), j))
        # with np.load(v_gt_path) as data:
        #     v_gt = data['x']
        #     v_gt_ = np.multiply(v_gt, obs_flag)
        #     mae_err = np.sum(np.abs(v_gt_))/vnum # mean absolute error
        #     mse_err = np.sum(v_gt_**2)/vnum # mean absolute error
        #     mae_gt_list.append(mae_err)
        #     mse_gt_list.append(mse_err)


        # v_path = os.path.join('log/de/velocity/smoke3_obs11_buo4_f150/final/eval/p2_n150', '%s_2.npz' % str(p[i]))
        # with np.load(v_path) as data:
        #     x = data['v']
        
        # mae_list = []
        # mse_list = []
        # mae_gt_list = []
        # mse_gt_list = []
        # for j in range(args.num_frames):
        #     v = x[j,...]
        #     v_ = np.multiply(v, obs_flag)
        #     mae_err = np.sum(np.abs(v_))/vnum # mean absolute error
        #     mse_err = np.sum(v_**2)/vnum # mean absolute error
        #     mae_list.append(mae_err)
        #     mse_list.append(mse_err)

            # if j == 130:
            #     v_big = np.multiply(v, 1-obs_flag)
            #     v_big = np.multiply(v_big, obs_big_flag)                
            #     n = np.sum(np.abs(v_big))
            #     print(n)
            #     if i == 0:
            #         norm = n
            #     else:                    
            #         factor = float(norm)/n
            #         print(factor)

            # # debug
            # print(v.min(), v.max(), v_.min(), v_.max())
            # v_ = v_.transpose([2,0,1,3])
            # idx_mid = int(args.resolution_z*0.5)
            # plt.subplot(121)
            # plt.imshow(v_[idx_mid,:,:,0], origin='lower')
            # plt.subplot(122)
            # v = v.transpose([2,0,1,3])
            # plt.imshow(v[idx_mid,:,:,0], origin='lower')
            # plt.show()

        # plt.plot(range(args.num_frames), mae_list)
        # # plt.plot(range(args.num_frames), mse_list)
        # plt.show()

        save_dir = 'log/de/velocity/smoke3_obs11_buo4_f150/final/eval/p2_n150'
        mae_path = os.path.join(save_dir, '%s_2_mae.csv' % str(p[i]))
        np.savetxt(mae_path, mae_list, delimiter=',')
        mse_path = os.path.join(save_dir, '%s_2_mse.csv' % str(p[i]))
        np.savetxt(mse_path, mse_list, delimiter=',')
        mae_path = os.path.join(save_dir, '%s_2_gt_mae.csv' % str(p[i]))
        np.savetxt(mae_path, mae_gt_list, delimiter=',')
        # mse_path = os.path.join(save_dir, '%s_2_gt_mse.csv' % str(p[i]))
        # np.savetxt(mse_path, mse_gt_list, delimiter=',')

        # idx = np.argmax(mae_list)
        # print('max frame:', idx)
        idx = 80

        v = x[idx,...]
        v = v.transpose([2,0,1,3])
        v = v[np.newaxis,:,::-1,:,:]
        def jacobian_np3(x):
            # x: bzyxd
            dudx = x[:,:,:,1:,0] - x[:,:,:,:-1,0]
            dvdx = x[:,:,:,1:,1] - x[:,:,:,:-1,1]
            dwdx = x[:,:,:,1:,2] - x[:,:,:,:-1,2]
            dudy = x[:,:,:-1,:,0] - x[:,:,1:,:,0] # y fliped
            dvdy = x[:,:,:-1,:,1] - x[:,:,1:,:,1] # y fliped
            dwdy = x[:,:,:-1,:,2] - x[:,:,1:,:,2] # y fliped
            dudz = x[:,1:,:,:,0] - x[:,:-1,:,:,0]
            dvdz = x[:,1:,:,:,1] - x[:,:-1,:,:,1]
            dwdz = x[:,1:,:,:,2] - x[:,:-1,:,:,2]

            dudx = np.concatenate((dudx, np.expand_dims(dudx[:,:,:,-1], axis=3)), axis=3)
            dvdx = np.concatenate((dvdx, np.expand_dims(dvdx[:,:,:,-1], axis=3)), axis=3)
            dwdx = np.concatenate((dwdx, np.expand_dims(dwdx[:,:,:,-1], axis=3)), axis=3)

            dudy = np.concatenate((np.expand_dims(dudy[:,:,0,:], axis=2), dudy), axis=2)
            dvdy = np.concatenate((np.expand_dims(dvdy[:,:,0,:], axis=2), dvdy), axis=2)
            dwdy = np.concatenate((np.expand_dims(dwdy[:,:,0,:], axis=2), dwdy), axis=2)

            dudz = np.concatenate((dudz, np.expand_dims(dudz[:,-1,:,:], axis=1)), axis=1)
            dvdz = np.concatenate((dvdz, np.expand_dims(dvdz[:,-1,:,:], axis=1)), axis=1)
            dwdz = np.concatenate((dwdz, np.expand_dims(dwdz[:,-1,:,:], axis=1)), axis=1)

            u = dwdy - dvdz
            v = dudz - dwdx
            w = dvdx - dudy
            
            j = np.stack([dudx,dudy,dudz,dvdx,dvdy,dvdz,dwdx,dwdy,dwdz], axis=-1)
            c = np.stack([u,v,w], axis=-1)
            
            return j, c            
        _, c = jacobian_np3(v)
        c = np.sqrt(np.sum(c**2, axis=-1))[0]
        v = np.sqrt(np.sum(v**2, axis=-1))[0]
        idx_mid = int(args.resolution_z*0.5)
        # plt.subplot(121)
        # plt.imshow(v[idx_mid,:,:], cmap=plt.cm.afmhot)
        # plt.subplot(122)
        # plt.imshow(c[idx_mid,:,:], cmap=plt.cm.afmhot)
        # plt.show()

        def saveimg(x, filename):
            x /= x.max()
            x = np.uint8(plt.cm.afmhot(x)*255)
            from PIL import Image
            im = Image.fromarray(x)
            im.save(filename)

        vort_mag_path = os.path.join(save_dir, '%s_2_vort_mag_%d.png' % (str(p[i]), idx))
        saveimg(c[idx_mid,:,:], vort_mag_path)
        vel_mag_path = os.path.join(save_dir, '%s_2_vel_mag_%d.png' % (str(p[i]), idx))
        saveimg(v[idx_mid,:,:], vel_mag_path)

    v_gt_path = os.path.join(args.log_dir, 'smoke3_obs1_buo1_intp_f150', 'v', '0_0_80.npz')
    with np.load(v_gt_path) as data:
        v = data['x']
    v = v.transpose([2,0,1,3])
    v = v[np.newaxis,:,::-1,:,:]

    _, c = jacobian_np3(v)
    c = np.sqrt(np.sum(c**2, axis=-1))[0]
    v = np.sqrt(np.sum(v**2, axis=-1))[0]

    vort_mag_path = os.path.join(save_dir, '%s_2_vort_mag_%d_gt.png' % (str(p[i]), idx))
    saveimg(c[idx_mid,:,:], vort_mag_path)
    vel_mag_path = os.path.join(save_dir, '%s_2_vel_mag_%d_gt.png' % (str(p[i]), idx))
    saveimg(v[idx_mid,:,:], vel_mag_path)    
                
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

        # obs_radius = gs.x*args.obs_radius
        # for p0 in [0.44, 0.47, 0.5]:
        #     obs_center = gs*vec3(0.44,args.obs_y_pos,args.obs_z_pos)
        #     obs = s.create(Sphere, center=obs_center, radius=0.10)
        #     obs.applyToGrid(grid=vel, value=vec3(1,1,1))
        #     copyGridToArrayMAC(target=v_, _Source=vel)
        #     v_file_path = os.path.join(args.log_dir, 'obs', '%f.npz' % p0)
        #     np.savez_compressed(v_file_path, 
        #                         x=v_)

        #     obs = s.create(Sphere, center=obs_center, radius=obs_radius+0.3)
        #     obs.applyToGrid(grid=vel, value=vec3(1,1,1))
        #     copyGridToArrayMAC(target=v_, _Source=vel)
        #     v_file_path = os.path.join(args.log_dir, 'obs', '%f_big.npz' % p0)
        #     np.savez_compressed(v_file_path, 
        #                         x=v_)
        # exit()

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
    blend_test()
    # obs_penetration_test()
    # if args.is_test:
    #     test()
    # else:
    #     main()