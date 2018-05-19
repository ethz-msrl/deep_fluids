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

parser.add_argument("--log_dir", type=str, default='data/liquid3_d5_r10_f150')

parser.add_argument("--num_param", type=int, default=3)
parser.add_argument("--path_format", type=str, default='%d_%d_%d.npz')
parser.add_argument("--p0", type=str, default='dist')
parser.add_argument("--p1", type=str, default='rot')
parser.add_argument("--p2", type=str, default='frames')

parser.add_argument("--min_dist", type=float, default=0.15)
parser.add_argument("--max_dist", type=float, default=0.25)
parser.add_argument("--num_dist", type=int, default=5)
parser.add_argument("--min_rot", type=float, default=0)
parser.add_argument("--max_rot", type=float, default=162) # 180.0/n*(n-1), 5: 144, 10: 162
parser.add_argument("--num_rot", type=int, default=10)
parser.add_argument("--src_y_pos", type=float, default=0.6)
parser.add_argument("--src_radius", type=float, default=0.1)
parser.add_argument("--basin_y_pos", type=float, default=0.2)
parser.add_argument("--min_frames", type=int, default=0)
parser.add_argument("--max_frames", type=int, default=149)
parser.add_argument("--num_frames", type=int, default=150)
parser.add_argument("--num_simulations", type=int, default=7500)

parser.add_argument("--resolution_x", type=int, default=96)
parser.add_argument("--resolution_y", type=int, default=48)
parser.add_argument("--resolution_z", type=int, default=96)
parser.add_argument("--gravity", type=float, default=-1e-3)
parser.add_argument("--radius_factor", type=float, default=1)
parser.add_argument("--min_particles", type=int, default=3)
parser.add_argument("--bWidth", type=int, default=1)
parser.add_argument("--open_bound", type=bool, default=False)
parser.add_argument("--time_step", type=float, default=0.8)

parser.add_argument('--is_test', type=str2bool, default=False)
parser.add_argument('--is_slow', type=str2bool, default=False)
parser.add_argument('--vpath', type=str, default='')

args = parser.parse_args()


def get_param(p1, p2):
    min_p1 = args.min_dist
    max_p1 = args.max_dist
    num_p1 = args.num_dist
    min_p2 = args.min_rot
    max_p2 = args.max_rot
    num_p2 = args.num_rot
    p1_ = p1/(num_p1-1) * (max_p1-min_p1) + min_p1
    p2_ = p2/(num_p2-1) * (max_p2-min_p2) + min_p2
    return p1_, p2_

def test():
    filename = os.path.basename(args.vpath)[:-4]
    print(filename)

    p1, p2 = filename.split('_')
    p1, p2 = get_param(float(p1), float(p2))
    title = 'm_' + filename    
    pt_dir = os.path.join(os.path.dirname(args.vpath), filename, 'pt')
    if not os.path.exists(pt_dir):
        os.mkdir(pt_dir)
    
    # solver params
    res_x = args.resolution_x
    res_y = args.resolution_y
    res_z = args.resolution_z
    gs = vec3(res_x, res_y, res_z)
    gravity = vec3(0,args.gravity,0)

    s = Solver(name='main', gridSize=gs, dim=3)
    s.timestep = args.time_step

    flags = s.create(FlagGrid)
    vel = s.create(MACGrid)

    pp = s.create(BasicParticleSystem) 
    # mesh = s.create(Mesh)

    # acceleration data for particle nbs
    pindex = s.create(ParticleIndexSystem) 
    gpi = s.create(IntGrid)

    flags.initDomain(boundaryWidth=args.bWidth)
    vel.clear()
    pp.clear()
    pindex.clear()
    gpi.clear()

    fluidBasin = Box(parent=s, p0=gs*vec3(0,0,0), p1=gs*vec3(1.0,args.basin_y_pos,1.0)) # basin
    fluidCenter = gs*vec3(0.5,args.src_y_pos,0.5)

    r = gs.x*p1
    th1 = p2/180.0*np.pi
    th2 = th1 + np.pi
    c1 = fluidCenter + vec3(r*np.cos(th1),0,r*np.sin(th1))
    c2 = fluidCenter + vec3(r*np.cos(th2),0,r*np.sin(th2))
    print(p1, p2, c1, c2)
		
    fluidDrop1 = Sphere(parent=s, center=c1, radius=gs.x*args.src_radius)
    fluidDrop2 = Sphere(parent=s, center=c2, radius=gs.x*args.src_radius)

    phi = fluidBasin.computeLevelset()
    phi.join(fluidDrop1.computeLevelset())
    phi.join(fluidDrop2.computeLevelset())

    flags.updateFromLevelset(phi)
    sampleLevelsetWithParticles(phi=phi, flags=flags, parts=pp, discretization=2, randomness=0.05)

    test = True
    if test:
        with np.load(args.vpath) as data:
            x = data['v']
            # x = data['x']

    intv = args.num_frames
    if args.is_slow:
        startf, endf = 24, 29
        denom = 10
        intv2 = (endf-startf+1)*denom
        intv = args.num_frames - (endf-startf+1) + intv2        

    for i in range(intv):
        # load extrapolated velocity
        if test:
            v = x[i,...]
        else:
            vpath = vel_path[:-4] + '_%d.npz' % i
            with np.load(vpath) as data:
                v = data['x']
        copyArrayToGridMAC(target=vel, source=v)

        if args.is_slow:
            if i >= startf and i < startf+intv2:
                s.timestep = args.time_step / float(denom)
            else:
                s.timestep = args.time_step

        markFluidCells(parts=pp, flags=flags)
        # if i > 100:
        #     checkHang(parts=m.pp, vel=m.vel, flags=m.flags, threshold=0.01) # 0.05
        extrapolateMACSimple(flags=flags, vel=vel, distance=4)

        gridParticleIndex(parts=pp, flags=flags, indexSys=pindex, index=gpi)
        unionParticleLevelset(pp, pindex, flags, gpi, phi, args.radius_factor)
        # averagedParticleLevelset(pp, pindex, flags, gpi, phi, args.radius_factor, 1, 1)
        # phi.setBound(1, boundaryWidth=args.bWidth)
        resetOutflow(flags=flags, parts=pp, index=gpi, indexSys=pindex)

        # extrapolate levelset, needed by particle resampling in adjustNumber / resample
        extrapolateLsSimple(phi=phi, distance=4, inside=True)

        
        # set source grids for resampling, used in adjustNumber!
        adjustNumber(parts=pp, vel=vel, flags=flags, minParticles=args.min_particles,
                     maxParticles=2*args.min_particles, phi=phi, radiusFactor=args.radius_factor)

        
        pp.advectInGrid(flags=flags, vel=vel, integrationMode=IntRK4, deleteInObstacle=False)
        
        # create mesh for vis.
        # phi.createMesh(mesh)
        # for iters in range(5):
        #     smoothMesh(mesh=m.mesh, strength=1e-3, steps=10) 
        #     subdivideMesh(mesh=m.mesh, minAngle=0.01, minLength=0.5, maxLength=3*0.5, cutTubes=True)
        
        # obj_path = os.path.join(obj_dir, title+'_%04d.obj' % i)
        # m.mesh.save(obj_path)
        
        pt_path = os.path.join(pt_dir, title+'_%04d.uni' % i)
        pp.save(pt_path)

        s.step()

def main():
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    field_type = ['v', 'obj', 'pt'] # , 'p', 's']
    for field in field_type:
        field_path = os.path.join(args.log_dir,field)
        if not os.path.exists(field_path):
            os.mkdir(field_path)

    args_file = os.path.join(args.log_dir, 'args.txt')
    with open(args_file, 'w') as f:
        for k, v in vars(args).items():
            print('  %s: %s' % (k, v))
            f.write('%s: %s\n' % (k, v))

    p1_space = np.linspace(args.min_dist, 
                            args.max_dist,
                            args.num_dist)
    p2_space = np.linspace(args.min_rot,
                            args.max_rot,
                            args.num_rot)
    p_list = np.array(np.meshgrid(p1_space, p2_space)).T.reshape(-1, 2)
    pi1_space = range(args.num_dist)
    pi2_space = range(args.num_rot)
    pi_list = np.array(np.meshgrid(pi1_space, pi2_space)).T.reshape(-1, 2)

    # solver params
    res_x = args.resolution_x
    res_y = args.resolution_y
    res_z = args.resolution_z
    gs = vec3(res_x, res_y, res_z)
    gravity = vec3(0,args.gravity,0)

    s = Solver(name='main', gridSize=gs, dim=3)
    s.timestep = args.time_step

    flags = s.create(FlagGrid)
    vel = s.create(MACGrid)
    pressure = s.create(RealGrid)

    # flip
    velOld = s.create(MACGrid)
    tmpVec3 = s.create(VecGrid)

    pp = s.create(BasicParticleSystem) 
    pVel = pp.create(PdataVec3)
    mesh = s.create(Mesh)

    # acceleration data for particle nbs
    pindex = s.create(ParticleIndexSystem) 
    gpi = s.create(IntGrid)

    # omega = s.create(VecGrid)
    # stream = s.create(VecGrid)
    # # vel_out = s.create(MACGrid)

    v_ = np.zeros([res_y,res_x,res_z,3], dtype=np.float32)
    # l_ = np.zeros([res_z,res_y,res_x], dtype=np.float32)
    # p_ = np.zeros([res_z,res_y,res_x], dtype=np.float32)
    # s_ = np.zeros([res_y,res_x,res_z,3], dtype=np.float32)

    if (GUI):
        gui = Gui()
        gui.show(True)
        gui.nextVec3Display()
        gui.nextVec3Display()
        gui.nextVec3Display()
        #gui.pause()

    print('start generation')
    v_range = [np.finfo(np.float).max, np.finfo(np.float).min]
    # l_range = [np.finfo(np.float).max, np.finfo(np.float).min]
    # p_range = [np.finfo(np.float).max, np.finfo(np.float).min]
    # s_range = [np.finfo(np.float).max, np.finfo(np.float).min]
    for i in trange(len(p_list), desc='scenes'):
        flags.initDomain(boundaryWidth=args.bWidth)
        # if args.open_bound:
        # 	setOpenBound(flags, args.bWidth,'xXyY', FlagOutflow|FlagEmpty)

        vel.clear()
        pressure.clear()
        # stream.clear()
        
        velOld.clear()
        tmpVec3.clear()

        pp.clear()
        pVel.clear()

        # scene setup
        fluidBasin = Box(parent=s, p0=gs*vec3(0,0,0), p1=gs*vec3(1.0,args.basin_y_pos,1.0)) # basin
        fluidCenter = gs*vec3(0.5,args.src_y_pos,0.5)

        p0, p1 = p_list[i][0], p_list[i][1]
        r = gs.x*p0
        th1 = p1/180.0*np.pi
        th2 = th1 + np.pi
        c1 = fluidCenter + vec3(r*np.cos(th1),0,r*np.sin(th1))
        c2 = fluidCenter + vec3(r*np.cos(th2),0,r*np.sin(th2))
        print(p, c1, c2)
        
        fluidDrop1 = Sphere(parent=s, center=c1, radius=gs.x*args.src_radius)
        fluidDrop2 = Sphere(parent=s, center=c2, radius=gs.x*args.src_radius)

        fluidVel1 = Sphere(parent=s, center=c1, radius=gs.x*(args.src_radius+0.05))
        fluidVel2 = Sphere(parent=s, center=c2, radius=gs.x*(args.src_radius+0.05))
        fluidSetVel = vec3(0,-1,0)

        phi = fluidBasin.computeLevelset()
        phi.join(fluidDrop1.computeLevelset())
        phi.join(fluidDrop2.computeLevelset())

        flags.updateFromLevelset(phi)
        sampleLevelsetWithParticles(phi=phi, flags=flags, parts=pp, discretization=2, randomness=0.05)

        # set initial velocity
        fluidVel1.applyToGrid(grid=vel, value=fluidSetVel)
        fluidVel2.applyToGrid(grid=vel, value=fluidSetVel)
        mapGridToPartsVec3(source=vel, parts=pp, target=pVel)

        for t in trange(args.num_frames, desc='sim'):
            # FLIP 
            pp.advectInGrid(flags=flags, vel=vel, integrationMode=IntRK4, deleteInObstacle=False)

            # make sure we have velocities throught liquid region
            mapPartsToMAC(vel=vel, flags=flags, velOld=velOld, parts=pp, partVel=pVel, weight=tmpVec3) 
            extrapolateMACFromWeight(vel=vel, distance=2, weight=tmpVec3)  # note, tmpVec3 could be free'd now...
            markFluidCells(parts=pp, flags=flags)

            # create approximate surface level set, resample particles
            gridParticleIndex(parts=pp , flags=flags, indexSys=pindex, index=gpi)
            unionParticleLevelset(pp, pindex, flags, gpi, phi, args.radius_factor)
            # averagedParticleLevelset(pp, pindex, flags, gpi, phi, args.radius_factor, 1, 1)
            # phi.setBound(1, boundaryWidth=args.bWidth)
            resetOutflow(flags=flags, parts=pp, index=gpi, indexSys=pindex) 
            
            # copyGridToArrayLevelset(target=l_, source=phi)
            
            # extend levelset somewhat, needed by particle resampling in adjustNumber
            extrapolateLsSimple(phi=phi, distance=4, inside=True)

            # forces & pressure solve
            addGravity(flags=flags, vel=vel, gravity=gravity)
            setWallBcs(flags=flags, vel=vel)	
            solvePressure(flags=flags, vel=vel, pressure=pressure, phi=phi)
            setWallBcs(flags=flags, vel=vel)
            # copyGridToArrayReal(target=p_, source=pressure)

            # set source grids for resampling, used in adjustNumber!
            pVel.setSource(vel, isMAC=True)
            adjustNumber(parts=pp, vel=vel, flags=flags, minParticles=args.min_particles, maxParticles=2*args.min_particles, phi=phi, radiusFactor=args.radius_factor)

            # make sure we have proper velocities
            extrapolateMACSimple(flags=flags, vel=vel)
            flipVelocityUpdate(vel=vel, velOld=velOld, flags=flags, parts=pp, partVel=pVel, flipRatio=0.97)
            copyGridToArrayMAC(target=v_, _Source=vel)

            # # create mesh for vis.
            # phi.createMesh(mesh)
            # for iters in range(5):
            # 	smoothMesh(mesh=mesh, strength=1e-3, steps=10) 
            # 	subdivideMesh(mesh=mesh, minAngle=0.01, minLength=0.5, maxLength=3*0.5, cutTubes=True)

            # getStreamfunction(flags=flags, vel=vel, grid=stream)
            # copyGridToArrayReal(target=s_, source=stream)			

            v_range = [np.minimum(v_range[0], v_.min()),
                        np.maximum(v_range[1], v_.max())]
            # l_range = [np.minimum(l_range[0], l_.min()),
            # 		   np.maximum(l_range[1], l_.max())]
            # p_range = [np.minimum(p_range[0], p_.min()),
            # 		   np.maximum(p_range[1], p_.max())]
            # s_range = [np.minimum(s_range[0], s_.min()),
            # 		   np.maximum(s_range[1], s_.max())]

            pit = tuple(pi_list[i].tolist() + [t])
            param_ = p_list[i].tolist() + [t]
            v_file_path = os.path.join(args.log_dir, 'v', args.path_format % pit)
            np.savez_compressed(v_file_path, 
                                x=v_, # yxzd 
                                y=param_)

            # save particles
            pt_file_path = os.path.join(args.log_dir, 'pt', '%d_%d_%d.uni' % pit)
            pp.save(pt_file_path)

            # l_file_path = os.path.join(args.log_dir, 'l', args.path_format % pit)
            # np.savez_compressed(l_file_path, 
            # 					x=np.expand_dims(l_, axis=-1),
            # 					y=param_)

            # obj_file_path = os.path.join(args.log_dir, 'obj', '%.2e_%.2e_%d.obj' % tuple(param_))
            # mesh.save(obj_file_path)

            # p_file_path = os.path.join(args.log_dir, 'p', args.path_format % pit)
            # np.savez_compressed(p_file_path, 
            # 					x=np.expand_dims(p_, axis=-1),
            # 					y=param_)

            # s_file_path = os.path.join(args.log_dir, 's', args.path_format % pit)
            # np.savez_compressed(s_file_path, 
            # 					x=s_, # yxzd
            # 					y=param_)

            s.step()
        gc.collect()

    vrange_file = os.path.join(args.log_dir, 'v_range.txt')
    with open(vrange_file, 'w') as f:
        print('velocity min %.3f max %.3f' % (v_range[0], v_range[1]))
        f.write('%.3f\n' % v_range[0])
        f.write('%.3f' % v_range[1])

    # lrange_file = os.path.join(args.log_dir, 'l_range.txt')
    # with open(lrange_file, 'w') as f:
    #     print('%s: levelset min %.3f max %.3f' % (datetime.now(), l_range[0], l_range[1]))
    #     f.write('%.3f\n' % l_range[0])
    #     f.write('%.3f' % l_range[1])

    # prange_file = os.path.join(args.log_dir, 'p_range.txt')
    # with open(prange_file, 'w') as f:
    # 	print('%s: pressure min %.3f max %.3f' % (datetime.now(), p_range[0], p_range[1]))
    # 	f.write('%.3f\n' % p_range[0])
    # 	f.write('%.3f' % p_range[1])

    # srange_file = os.path.join(args.log_dir, 's_range.txt')
    # with open(srange_file, 'w') as f:
    # 	print('%s: stream min %.3f max %.3f' % (datetime.now(), s_range[0], s_range[1]))
    # 	f.write('%.3f\n' % s_range[0])
    # 	f.write('%.3f' % s_range[1])

    print('Done')


if __name__ == '__main__':
    if args.is_test:
        test()
    else:
        main()