import os
import numpy as np
import argparse

from utils import prepare_dirs_and_logger, save_image, convert_png2mp4
from ops import plane_view_np

from manta import *

try:
    from manta import *
except:
    print('no manta')
    pass

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default='log/de/velocity/liquid3d_d5_r5_f150/p2_n150')
parser.add_argument("--num_samples", type=int, default=150)
parser.add_argument("--gui_on", type=bool, default=False)
parser.add_argument("--data_type", type=str, default='velocity')
parser.add_argument("--resolution_x", type=int, default=96)
parser.add_argument("--resolution_y", type=int, default=48)
parser.add_argument("--resolution_z", type=int, default=96)
parser.add_argument("--gravity", type=float, default=-1e-3)
parser.add_argument("--radius_factor", type=float, default=1)
parser.add_argument("--min_particles", type=int, default=3)
parser.add_argument("--bWidth", type=int, default=1)
parser.add_argument("--open_bound", type=bool, default=False)
parser.add_argument("--time_step", type=float, default=0.8)

parser.add_argument("--min_dist", type=float, default=0.15)
parser.add_argument("--max_dist", type=float, default=0.25)
parser.add_argument("--num_dist", type=int, default=5)
parser.add_argument("--min_rot", type=float, default=0)
parser.add_argument("--max_rot", type=float, default=144) # 5: 144, 6: 162
parser.add_argument("--num_rot", type=int, default=5)
parser.add_argument("--src_y_pos", type=float, default=0.6)
parser.add_argument("--src_radius", type=float, default=0.1)
parser.add_argument("--basin_y_pos", type=float, default=0.2)
parser.add_argument("--min_frames", type=int, default=0)
parser.add_argument("--max_frames", type=int, default=149)
parser.add_argument("--num_frames", type=int, default=150)
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

def advect(vel_path, m, title, p1, p2):
    pt_dir = os.path.join(args.model_dir, 'pt')
    if not os.path.exists(pt_dir):
        os.makedirs(pt_dir)
    
    m.flags.initDomain(boundaryWidth=args.bWidth)
    m.vel.clear()
    m.pp.clear()
    m.pVel.clear()
    m.pindex.clear()
    m.gpi.clear()

    fluidBasin = Box(parent=m.s, p0=m.gs*vec3(0,0,0), p1=m.gs*vec3(1.0,args.basin_y_pos,1.0)) # basin
    fluidCenter = m.gs*vec3(0.5,args.src_y_pos,0.5)

    r = m.gs.x*p1
    th1 = p2/180.0*np.pi
    th2 = th1 + np.pi
    c1 = fluidCenter + vec3(r*np.cos(th1),0,r*np.sin(th1))
    c2 = fluidCenter + vec3(r*np.cos(th2),0,r*np.sin(th2))
    print(p1, p2, c1, c2)
		
    fluidDrop1 = Sphere(parent=m.s, center=c1, radius=m.gs.x*args.src_radius)
    fluidDrop2 = Sphere(parent=m.s, center=c2, radius=m.gs.x*args.src_radius)

    phi = fluidBasin.computeLevelset()
    phi.join(fluidDrop1.computeLevelset())
    phi.join(fluidDrop2.computeLevelset())

    m.flags.updateFromLevelset(phi)
    sampleLevelsetWithParticles(phi=phi, flags=m.flags, parts=m.pp, discretization=2, randomness=0.05)

    intv = args.num_samples
    test = True
    if test:
        with np.load(vel_path) as data:
            x = data['x']

    for i in range(intv):
        # load extrapolated velocity
        if test:
            v = x[i,...]
        else:
            vpath = vel_path[:-4] + '_%d.npz' % i
            with np.load(vpath) as data:
                v = data['x']
        copyArrayToGridMAC(target=m.vel, source=v)


        markFluidCells(parts=m.pp, flags=m.flags)
        checkHang(parts=m.pp, vel=m.vel, flags=m.flags, threshold=0.05)
        extrapolateMACSimple(flags=m.flags, vel=m.vel, distance=4)

        gridParticleIndex(parts=m.pp, flags=m.flags, indexSys=m.pindex, index=m.gpi)
        unionParticleLevelset(m.pp, m.pindex, m.flags, m.gpi, phi, args.radius_factor)
        # averagedParticleLevelset(m.pp, m.pindex, m.flags, m.gpi, phi, args.radius_factor, 1, 1)
        # phi.setBound(1, boundaryWidth=args.bWidth)
        resetOutflow(flags=m.flags, parts=m.pp, index=m.gpi, indexSys=m.pindex)

        # extrapolate levelset, needed by particle resampling in adjustNumber / resample
        extrapolateLsSimple(phi=phi, distance=4, inside=True)

        
        # set source grids for resampling, used in adjustNumber!
        adjustNumber(parts=m.pp, vel=m.vel, flags=m.flags, minParticles=args.min_particles,
                     maxParticles=2*args.min_particles, phi=phi, radiusFactor=args.radius_factor)

        
        m.pp.advectInGrid(flags=m.flags, vel=m.vel, integrationMode=IntRK4, deleteInObstacle=False)
        
        # create mesh
        phi.createMesh(m.mesh)
        # for iters in range(5):
        #     smoothMesh(mesh=m.mesh, strength=1e-3, steps=10) 
        #     subdivideMesh(mesh=m.mesh, minAngle=0.01, minLength=0.5, maxLength=3*0.5, cutTubes=True)
        
        # obj_path = os.path.join(obj_dir, title+'_%04d.obj' % i)
        # m.mesh.save(obj_path)
        
        pt_path = os.path.join(pt_dir, title+'_%04d.uni' % i)
        m.pp.save(pt_path)

        m.s.step()

def set_manta():
    class MantaObj(object):
        pass
    m = MantaObj()
    res_x = args.resolution_x
    res_y = args.resolution_y
    res_z = args.resolution_z
    m.gs = vec3(res_x, res_y, res_z)

    m.s = Solver(name='main', gridSize=m.gs, dim=3)
    frame_ratio = args.num_frames / args.num_samples
    m.s.timestep = args.time_step * frame_ratio
        
    m.flags = m.s.create(FlagGrid)
    m.vel = m.s.create(MACGrid)
    m.pp = m.s.create(BasicParticleSystem) 
    m.pVel = m.pp.create(PdataVec3)
    m.mesh = m.s.create(Mesh)
    
    # acceleration data for particle nbs
    m.pindex = m.s.create(ParticleIndexSystem)
    m.gpi = m.s.create(IntGrid)
        
    if (GUI and args.gui_on):
        gui = Gui()
        gui.show(True)
        # gui.pause()

    return m

def main():
    m = set_manta()

    # p1 = int(args.num_dist/2)-1
    # p2 = int(args.num_rot/2)-1
    # p12s = [
    #     [p1, p2], [p1, p2+0.5], [p1, p2+1],
    #     [p1+0.5, p2], [p1+0.5, p2+0.5], [p1+0.5, p2+1],
    #     [p1+1, p2], [p1+1, p2+0.5], [p1+1, p2+1],
    # ]

    # p12s = [[0,0]]

    p12s = [
        [4,0],[4,5],
    ]
        
    for p12 in p12s:
        p1, p2 = p12[0], p12[1]
        vel_path = os.path.join(args.model_dir, '{}_{}.npz'.format(p1, p2))
        title = 'd_{}_{}'.format(p1, p2)
        p1_, p2_ = get_param(p1, p2)
        advect(vel_path, m, title, p1_, p2_)

if __name__ == "__main__":
    main()