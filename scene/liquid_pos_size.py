import argparse
import sys
from datetime import datetime
import time
import os

import numpy as np
import matplotlib.pyplot as plt

import gc
try:
	from manta import *
except ImportError:
	pass

parser = argparse.ArgumentParser()

parser.add_argument("--log_dir", type=str, default='data/liquid_pos10_size4_f200')

parser.add_argument("--num_param", type=int, default=3)
parser.add_argument("--path_format", type=str, default='%d_%d_%d.npz')
parser.add_argument("--p0", type=str, default='src_x_pos')
parser.add_argument("--p1", type=str, default='src_radius')
parser.add_argument("--p2", type=str, default='frames')

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


def main():
	if not os.path.exists(args.log_dir):
		os.mkdir(args.log_dir)

	field_type = ['v', 's'] # 'p'
	for field in field_type:
		field_path = os.path.join(args.log_dir,field)
		if not os.path.exists(field_path):
			os.mkdir(field_path)
	
	args_file = os.path.join(args.log_dir, 'args.txt')
	with open(args_file, 'w') as f:
		print('%s: arguments' % datetime.now())
		for k, v in vars(args).items():
			print('  %s: %s' % (k, v))
			f.write('%s: %s\n' % (k, v))

	p1_space = np.linspace(args.min_src_x_pos, 
						   args.max_src_x_pos,
						   args.num_src_x_pos)
	p2_space = np.linspace(args.min_src_radius,
						   args.max_src_radius,
						   args.num_src_radius)
	p_list = np.array(np.meshgrid(p1_space, p2_space)).T.reshape(-1, 2)
	pi1_space = range(args.num_src_x_pos)
	pi2_space = range(args.num_src_radius)
	pi_list = np.array(np.meshgrid(pi1_space, pi2_space)).T.reshape(-1, 2)

	# solver params
	res_x = args.resolution_x
	res_y = args.resolution_y
	gs = vec3(res_x, res_y, 1)
	gravity = vec3(0,args.gravity,0)

	s = Solver(name='main', gridSize=gs, dim=2)
	s.timestep = args.time_step
	
	flags = s.create(FlagGrid)
	vel = s.create(MACGrid)
	pressure = s.create(RealGrid)
	# stream = s.create(RealGrid)

	# flip
	velOld = s.create(MACGrid)
	tmpVec3 = s.create(VecGrid)
	
	pp = s.create(BasicParticleSystem) 
	pVel = pp.create(PdataVec3)
	# mesh = s.create(Mesh)
	
	# acceleration data for particle nbs
	pindex = s.create(ParticleIndexSystem) 
	gpi = s.create(IntGrid)

	l_ = np.zeros([res_y,res_x], dtype=np.float32)
	p_ = np.zeros([res_y,res_x], dtype=np.float32)
	s_ = np.zeros([res_y,res_x], dtype=np.float32)
	v_ = np.zeros([res_y,res_x,3], dtype=np.float32)

	if (GUI):
		gui = Gui()
		gui.show(True)
		#gui.pause()

	print('start generation')
	sim_id = 0
	num_total_p = p_list.shape[0]
	num_total_sim = num_total_p * args.num_frames
	l_range = [np.finfo(np.float).max, np.finfo(np.float).min]
	v_range = [np.finfo(np.float).max, np.finfo(np.float).min]
	p_range = [np.finfo(np.float).max, np.finfo(np.float).min]
	s_range = [np.finfo(np.float).max, np.finfo(np.float).min]
	for i, (p, pi) in enumerate(zip(p_list,pi_list)):
		start_time = time.time()
		
		flags.initDomain(boundaryWidth=args.bWidth)
		if args.open_bound:
			setOpenBound(flags, args.bWidth,'xXyY', FlagOutflow|FlagEmpty)

		vel.clear()
		pressure.clear()
		# stream.clear()
		
		velOld.clear()
		tmpVec3.clear()
	
		pp.clear()
		pVel.clear()

		fluidBasin = Box(parent=s, p0=gs*vec3(0,0,0), p1=gs*vec3(1.0,args.basin_y_pos,1.0)) # basin
		dropCenter = vec3(p[0],args.src_y_pos,0.5)
		dropRadius = p[1]
		fluidDrop = Sphere(parent=s, center=gs*dropCenter, radius=gs.x*dropRadius)
		phi = fluidBasin.computeLevelset()
		phi.join(fluidDrop.computeLevelset())

		flags.updateFromLevelset(phi)
		sampleLevelsetWithParticles(phi=phi, flags=flags, parts=pp, discretization=2, randomness=0.05)

		fluidVel = Sphere(parent=s, center=gs*dropCenter, radius=gs.x*(dropRadius+0.05))
		fluidSetVel = vec3(0,-1,0)
		
		# set initial velocity
		fluidVel.applyToGrid(grid=vel, value=fluidSetVel)
		mapGridToPartsVec3(source=vel, parts=pp, target=pVel)

		for t in range(args.num_frames):
			print('%s: simulating %d of %d (%d/%d)' % (datetime.now(), sim_id, num_total_sim, i+1, num_total_p))

			# FLIP 
			pp.advectInGrid(flags=flags, vel=vel, integrationMode=IntRK4, deleteInObstacle=False)
			# make sure we have velocities throught liquid region
			mapPartsToMAC(vel=vel, flags=flags, velOld=velOld, parts=pp, partVel=pVel, weight=tmpVec3) 
			extrapolateMACFromWeight(vel=vel, distance=2, weight=tmpVec3)  # note, tmpVec3 could be free'd now...
			markFluidCells(parts=pp, flags=flags)

			# create approximate surface level set, resample particles
			gridParticleIndex(parts=pp , flags=flags, indexSys=pindex, index=gpi)
			unionParticleLevelset(pp, pindex, flags, gpi, phi, args.radius_factor) 
			resetOutflow(flags=flags, parts=pp, index=gpi, indexSys=pindex) 
			copyGridToArrayLevelset(target=l_, source=phi)
			
			# extend levelset somewhat, needed by particle resampling in adjustNumber
			extrapolateLsSimple(phi=phi, distance=4, inside=True); 

			# forces & pressure solve
			addGravity(flags=flags, vel=vel, gravity=gravity)
			setWallBcs(flags=flags, vel=vel)	
			solvePressure(flags=flags, vel=vel, pressure=pressure, phi=phi)
			setWallBcs(flags=flags, vel=vel)
			copyGridToArrayReal(target=p_, source=pressure)

			# set source grids for resampling, used in adjustNumber!
			pVel.setSource(vel, isMAC=True)
			adjustNumber(parts=pp, vel=vel, flags=flags, minParticles=args.min_particles, maxParticles=2*args.min_particles, phi=phi, radiusFactor=args.radius_factor)

			# save before extrapolation
			copyGridToArrayMAC(target=v_, _Source=vel)

			# make sure we have proper velocities
			extrapolateMACSimple(flags=flags, vel=vel, distance=4)
			flipVelocityUpdate(vel=vel, velOld=velOld, flags=flags, parts=pp, partVel=pVel, flipRatio=0.97)
			
			# save after extrapolation
			# copyGridToArrayMAC(target=v_, _Source=vel)
			
			# getStreamfunction(flags=flags, vel=vel, grid=stream)
			# copyGridToArrayReal(target=s_, source=stream)
			
			param_ = [p[0], p[1], t]			
			
			l_range = [np.minimum(l_range[0], l_.min()),
					   np.maximum(l_range[1], l_.max())]
			v_range = [np.minimum(v_range[0], v_.min()),
					   np.maximum(v_range[1], v_.max())]
			p_range = [np.minimum(p_range[0], p_.min()),
					   np.maximum(p_range[1], p_.max())]
			s_range = [np.minimum(s_range[0], s_.min()),
					   np.maximum(s_range[1], s_.max())]
			
			v_file_path = os.path.join(args.log_dir, 'v', args.path_format % (pi[0], pi[1], t))
			np.savez_compressed(v_file_path, 
								x=v_[:,:,:2],
								y=param_)

			# sim_file_path = os.path.join(args.log_dir, 'sim_' + args.path_format % (pi[0], pi[1], t))
			# np.savez_compressed(sim_file_path, 
			# 					l=np.expand_dims(l_, axis=-1),
			# 					p=np.expand_dims(p_, axis=-1),
			# 					s=np.expand_dims(s_, axis=-1),
			# 					v=v_[:,:,:2],
			# 					param=param_)

			s.step()
			sim_id += 1
			# break
		
		gc.collect()
		duration = time.time() - start_time
		print('%s: done %d of %d (%.3f sec)' % (datetime.now(), sim_id, num_total_sim, duration))

	vrange_file = os.path.join(args.log_dir, 'v_range.txt')
	with open(vrange_file, 'w') as f:
		print('%s: velocity min %.3f max %.3f' % (datetime.now(), v_range[0], v_range[1]))
		f.write('%.3f\n' % v_range[0])
		f.write('%.3f' % v_range[1])

	# lrange_file = os.path.join(args.log_dir, 'l_range.txt')
	# with open(lrange_file, 'w') as f:
	# 	print('%s: levelset min %.3f max %.3f' % (datetime.now(), l_range[0], l_range[1]))
	# 	f.write('%.3f\n' % l_range[0])
	# 	f.write('%.3f' % l_range[1])

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
    main()