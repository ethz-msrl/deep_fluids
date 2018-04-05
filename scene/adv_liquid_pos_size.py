import os
import numpy as np
import argparse

from manta import *


try:
    from manta import *
except:
    print('no manta')
    pass

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default='log/de/velocity/liquid_pos10_size5_f200/p2_n200')
parser.add_argument("--num_samples", type=int, default=200)
parser.add_argument("--gui_on", type=bool, default=False)
parser.add_argument("--data_type", type=str, default='velocity')

parser.add_argument("--num_src_x_pos", type=int, default=10)
parser.add_argument("--min_src_x_pos", type=float, default=0.2)
parser.add_argument("--max_src_x_pos", type=float, default=0.8)
parser.add_argument("--src_y_pos", type=float, default=0.6)
parser.add_argument("--num_src_radius", type=int, default=4)
parser.add_argument("--min_src_radius", type=float, default=0.04)
parser.add_argument("--max_src_radius", type=float, default=0.08)
parser.add_argument("--basin_y_pos", type=float, default=0.2)
parser.add_argument("--num_frames", type=int, default=200)
parser.add_argument("--min_frames", type=int, default=0)
parser.add_argument("--max_frames", type=int, default=199)
parser.add_argument("--num_simulations", type=int, default=8000)

parser.add_argument("--resolution_x", type=int, default=128)
parser.add_argument("--resolution_y", type=int, default=64)
parser.add_argument("--gravity", type=float, default=-1e-3)
parser.add_argument("--radius_factor", type=float, default=1)
parser.add_argument("--min_particles", type=int, default=2)
parser.add_argument("--bWidth", type=int, default=1)
parser.add_argument("--open_bound", type=bool, default=False)
parser.add_argument("--time_step", type=float, default=0.5)

args = parser.parse_args()

def get_param(p1, p2):
    min_p1 = args.min_src_x_pos
    max_p1 = args.max_src_x_pos
    num_p1 = args.num_src_x_pos
    min_p2 = args.min_src_radius
    max_p2 = args.max_src_radius
    num_p2 = args.num_src_radius
    p1_ = p1/(num_p1-1) * (max_p1-min_p1) + min_p1
    p2_ = p2/(num_p2-1) * (max_p2-min_p2) + min_p2
    return p1_, p2_

def advect(vel_path, m, title, p1, p2):
    l_ = np.zeros([args.resolution_y,args.resolution_x], dtype=np.float32)
    img_dir = os.path.join(args.model_dir, title)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    
    m.flags.initDomain(boundaryWidth=args.bWidth)
    if m.open_bound:
        setOpenBound(m.flags, args.bWidth,'xXyY', FlagOutflow|FlagEmpty)

    m.vel.clear()
    m.pp.clear()
    m.pVel.clear()
    m.pindex.clear()
    m.gpi.clear()

    fluidBasin = Box(parent=m.s, p0=m.gs*vec3(0,0,0), p1=m.gs*vec3(1.0,args.basin_y_pos)*0.9,1.0)) # basin
    dropCenter = vec3(p1,args.src_y_pos),0.5)
    fluidDrop = Sphere(parent=m.s, center=m.gs*dropCenter, radius=m.gs.x*p2)

    phi = fluidBasin.computeLevelset()
    phi.join(fluidDrop.computeLevelset())
    # phi.join(fluidDropSmall.computeLevelset())

    m.flags.updateFromLevelset(phi)
    sampleLevelsetWithParticles(phi=phi, flags=m.flags, parts=m.pp, discretization=2, randomness=0.05)

    # phi.join(fluidDrop.computeLevelset())

    intv = args.num_samples

    for i in range(intv):
        # advect
        m.pp.advectInGrid(flags=m.flags, vel=m.vel, integrationMode=IntRK4, deleteInObstacle=False)

        # create surface
        markFluidCells(parts=m.pp, flags=m.flags)
        gridParticleIndex(parts=m.pp, flags=m.flags, indexSys=m.pindex, index=m.gpi)
        unionParticleLevelset(m.pp, m.pindex, m.flags, m.gpi, phi, args.radiusFactor) # faster, but not as smooth
        # averagedParticleLevelset(m.pp, m.pindex, m.flags, m.gpi, phi, radiusFactor, 1, 1)
        # phi.setBound(value=1, boundaryWidth=1)
        resetOutflow(flags=m.flags, parts=m.pp, index=m.gpi, indexSys=m.pindex) 

        # save levelset
        copyGridToArrayLevelset(target=l_, source=phi)
        lv = np.expand_dims(l_[::-1,:], axis=-1)
        # offset = 0.0
        # eps = 1e-3
        # lv[lv<(offset+eps)] = -1
        # lv[lv>-1] = 1
        lv /= config.l_range # [-1, 1]
        img = np.clip((lv+1)*127.5, 0, 255) # [0,255]

        img_path = os.path.join(img_dir, '%04d.png' % i)
        save_image(img, img_path, single=True)

        # load extrapolated velocity
        v = x[i,::-1,:,:]
        v = np.dstack((v,np.zeros([config.height,config.width,1])))
        copyArrayToGridMAC(target=m.vel, source=v)

        # extrapolate levelset, needed by particle resampling in adjustNumber / resample
        extrapolateLsSimple(phi=phi, distance=4, inside=True)
        adjustNumber(parts=m.pp, vel=m.vel, flags=m.flags, 
                     minParticles=args.minParticles, maxParticles=2*args.minParticles,
                     phi=phi, radiusFactor=args.radiusFactor)

        m.s.step()

    mp4_path = os.path.join(args.model_dir, title+'.mp4')
    fps = int(40 * intv / 200.0)
    convert_png2mp4(img_dir, mp4_path, fps, delete_imgdir=False)

def set_manta():
    class MantaObj(object):
        pass
    m = MantaObj()
    res_x = args.resolution_x
    res_y = args.resolution_y
    m.gs = vec3(res_x, res_y, 1)

    m.s = Solver(name='main', gridSize=m.gs, dim=2)
    frame_ratio = args.num_frames / args.num_samples
    m.s.timestep = args.time_step * frame_ratio
        
    m.flags = m.s.create(FlagGrid)
    m.vel = m.s.create(MACGrid)
    m.pp = m.s.create(BasicParticleSystem) 
    m.pVel = m.pp.create(PdataVec3)
    
    # acceleration data for particle nbs
    m.pindex = m.s.create(ParticleIndexSystem)
    m.gpi = m.s.create(IntGrid)
        
    if (GUI and args.gui_on):
        gui = Gui()
        gui.show(True)
        # gui.pause()

    return m

def main(config):
    m = set_manta()

    p12s = [
        [4,0],
    ]

    for p12 in p12s:
        p1, p2 = p12[0], p12[1]
        vel_path = os.path.join(args.model_dir, '{}_{}.npz'.format(p1, p2))
        title = 'd_{}_{}'.format(p1, p2)
        p1_, p2_ = get_param(p1, p2)
        advect(vel_path, m, title, p1_, p2_)


if __name__ == "__main__":
    main()