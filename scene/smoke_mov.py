import argparse
import sys
from datetime import datetime
import time
import os
from tqdm import trange

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from perlin import TileableNoise

try:
	from manta import *
	import gc
except ImportError:
	pass
else:
	pass

parser = argparse.ArgumentParser()

parser.add_argument("--log_dir", type=str, default='data/smoke_mov500_f200')

parser.add_argument("--num_param", type=int, default=2)
parser.add_argument("--path_format", type=str, default='%d_%d.npz')
parser.add_argument("--p0", type=str, default='sim_id')
parser.add_argument("--p1", type=str, default='frames')

num_s = 500
num_f = 200
num_sim = num_s*num_f
parser.add_argument("--min_src_x_pos", type=float, default=0.1)
parser.add_argument("--max_src_x_pos", type=float, default=0.9)
parser.add_argument("--src_y_pos", type=float, default=0.1)
parser.add_argument("--src_radius", type=float, default=0.08)
parser.add_argument("--num_scenes", type=int, default=num_s)
parser.add_argument("--num_frames", type=int, default=num_f)
parser.add_argument("--min_frames", type=int, default=0)
parser.add_argument("--max_frames", type=int, default=num_f-1)
parser.add_argument("--num_simulations", type=int, default=num_sim)

parser.add_argument("--resolution_x", type=int, default=96)
parser.add_argument("--resolution_y", type=int, default=128)
parser.add_argument("--resolution_z", type=int, default=1)
parser.add_argument("--buoyancy", type=float, default=-4e-3)
parser.add_argument("--bWidth", type=int, default=1)
parser.add_argument("--open_bound", type=str, default='xXyY')
parser.add_argument("--time_step", type=float, default=0.5)
parser.add_argument("--clamp_mode", type=int, default=2)

parser.add_argument("--nscale", type=float, default=0.05)
parser.add_argument("--nrepeat", type=int, default=1000)
parser.add_argument("--nseed", type=int, default=123)

args = parser.parse_args()


def nplot():
	n_path = os.path.join(args.log_dir, 'n.npz')
	with np.load(n_path) as data:
		n_list = data['n']

	t = range(args.num_frames)
	fig = plt.figure()
	plt.ylim((0,1))
	for i in range(args.num_scenes):
		plt.plot(t, n_list[i,:])

	n_fig_path = os.path.join(args.log_dir, 'n.png')
	fig.savefig(n_fig_path)	

def main():
	if not os.path.exists(args.log_dir):
		os.mkdir(args.log_dir)

	field_type = ['v'] #, 'vdb'] #, 's'] #, 'p']
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

	noise = TileableNoise(seed=args.nseed)

	# solver params
	res_x = args.resolution_x
	res_y = args.resolution_y
	res_z = args.resolution_z
	gs = vec3(res_x, res_y, res_z)
	
	if res_z > 1:
		s = Solver(name='main', gridSize=gs, dim=3)
	else:
		s = Solver(name='main', gridSize=gs, dim=2)
	s.timestep = args.time_step

	buoyancy = vec3(0,args.buoyancy,0)
	radius = gs.x*args.src_radius
	
	flags = s.create(FlagGrid)
	vel = s.create(MACGrid)
	density = s.create(RealGrid)
	pressure = s.create(RealGrid)
	# stream = s.create(RealGrid)

	if res_z > 1:
		v_ = np.zeros([res_y,res_x,res_z,3], dtype=np.float32)
		# d_ = np.zeros([res_z,res_y,res_x], dtype=np.float32)
		# s_ = np.zeros([res_y,res_x,res_z,3], dtype=np.float32)
		# p_ = np.zeros([res_z,res_y,res_x], dtype=np.float32)
	else:
		v_ = np.zeros([res_y,res_x,3], dtype=np.float32)
		# d_ = np.zeros([res_y,res_x], dtype=np.float32)
		# p_ = np.zeros([res_y,res_x], dtype=np.float32)
		# s_ = np.zeros([res_y,res_x], dtype=np.float32)

	if (GUI):
		gui = Gui()
		gui.show(True)
		gui.nextVec3Display()
		gui.nextVec3Display()
		gui.nextVec3Display()
		#gui.pause()

	print('start generation')
	sim_id = 0
	num_total_p = args.num_scenes
	num_total_sim = num_total_p * args.num_frames
	v_range = [np.finfo(np.float).max, np.finfo(np.float).min]
	# d_range = [np.finfo(np.float).max, np.finfo(np.float).min]
	# p_range = [np.finfo(np.float).max, np.finfo(np.float).min]
	# s_range = [np.finfo(np.float).max, np.finfo(np.float).min]
	n_list = []
	for i in trange(args.num_scenes, desc='scenes'):
		start_time = time.time()
		
		flags.initDomain(boundaryWidth=args.bWidth)
		flags.fillGrid()
		setOpenBound(flags, args.bWidth, args.open_bound, FlagOutflow|FlagEmpty)

		vel.clear()
		density.clear()
		pressure.clear()
		# stream.clear()
		
		# noise
		noise.randomize()
		ny = noise.rng.randint(args.num_frames)*args.nscale
		nz = noise.rng.randint(args.num_frames)*args.nscale
		nq = deque([-1]*args.num_frames,args.num_frames)
		
		for t in trange(args.num_frames, desc='sim', leave=False):
			# print('%s: simulating %d of %d (%d/%d)' % (datetime.now(), sim_id, num_total_sim, i+1, num_total_p))

			nx = noise.noise3(x=t*args.nscale, y=ny, z=nz, repeat=args.nrepeat)
			p = (nx+1)*0.5 * (args.max_src_x_pos-args.min_src_x_pos) + args.min_src_x_pos # [minx, maxx]
			nq.append(p)
			source = s.create(Sphere, center=gs*vec3(p,args.src_y_pos,0.5), radius=radius)

			source.applyToGrid(grid=density, value=1)
				
			advectSemiLagrange(flags=flags, vel=vel, grid=density, order=2,
							   openBounds=True, boundaryWidth=args.bWidth, clampMode=args.clamp_mode)
			advectSemiLagrange(flags=flags, vel=vel, grid=vel,     order=2,
							   openBounds=True, boundaryWidth=args.bWidth, clampMode=args.clamp_mode)
			setWallBcs(flags=flags, vel=vel)
			addBuoyancy(density=density, vel=vel, gravity=buoyancy, flags=flags)
			solvePressure(flags=flags, vel=vel, pressure=pressure, cgMaxIterFac=10.0, cgAccuracy=0.0001)
			setWallBcs(flags=flags, vel=vel)
			# getStreamfunction(flags=flags, vel=vel, grid=stream)
		
			copyGridToArrayMAC(target=v_, _Source=vel)
			# copyGridToArrayReal(target=d_, source=density)
			# copyGridToArrayReal(target=p_, source=pressure)
			# copyGridToArrayReal(target=s_, source=stream)
			
			param_ = list(nq)
			# nparam = [(pp - args.min_src_x_pos)/(args.max_src_x_pos-args.min_src_x_pos)*2 - 1 for pp in param_]
			# print(param_, nparam)

			v_range = [np.minimum(v_range[0], v_.min()),
					   np.maximum(v_range[1], v_.max())]
			# d_range = [np.minimum(d_range[0], d_.min()),
			# 		   np.maximum(d_range[1], d_.max())]
			# p_range = [np.minimum(p_range[0], p_.min()),
			# 		   np.maximum(p_range[1], p_.max())]
			# s_range = [np.minimum(s_range[0], s_.min()),
			# 		   np.maximum(s_range[1], s_.max())]

			v_file_path = os.path.join(args.log_dir, 'v', args.path_format % (i, t))
			np.savez_compressed(v_file_path,
								x=v_[:,:,:2], # x=v_, # yxzd for 3d
								y=param_)

			s.step()
			sim_id += 1

		n_list.append(param_)
		
		gc.collect()
		duration = time.time() - start_time

	n_path = os.path.join(args.log_dir, 'n.npz')
	np.savez_compressed(n_path, n=n_list)

	vrange_file = os.path.join(args.log_dir, 'v_range.txt')
	with open(vrange_file, 'w') as f:
		print('%s: velocity min %.3f max %.3f' % (datetime.now(), v_range[0], v_range[1]))
		f.write('%.3f\n' % v_range[0])
		f.write('%.3f' % v_range[1])

	# drange_file = os.path.join(args.log_dir, 'd_range.txt')
	# with open(drange_file, 'w') as f:
	# 	print('%s: density min %.3f max %.3f' % (datetime.now(), d_range[0], d_range[1]))
	# 	f.write('%.3f\n' % d_range[0])
	# 	f.write('%.3f' % d_range[1])

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

	nplot()

	print('Done')

if __name__ == '__main__':
	main()
	# nplot()