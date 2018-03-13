import argparse
import sys
from datetime import datetime
import time
import os

import numpy as np
import matplotlib.pyplot as plt

from manta import *
import gc

parser = argparse.ArgumentParser()

parser.add_argument("--log_dir", type=str, default='D:/Polybox/dev/deep-fluids/data/smoke_pos2_f100_test')

parser.add_argument("--num_param", type=int, default=2)
parser.add_argument("--path_format", type=str, default='%d_%d.npz')
parser.add_argument("--p0", type=str, default='src_x_pos')
parser.add_argument("--p1", type=str, default='frames')

num_p = 2
num_sim = num_p*100
parser.add_argument("--num_src_x_pos", type=int, default=num_p)
parser.add_argument("--min_src_x_pos", type=float, default=0.2)
parser.add_argument("--max_src_x_pos", type=float, default=0.8)
parser.add_argument("--src_y_pos", type=float, default=0.1)
parser.add_argument("--src_radius", type=float, default=0.08)
parser.add_argument("--num_frames", type=int, default=100)
parser.add_argument("--min_frames", type=int, default=0)
parser.add_argument("--max_frames", type=int, default=99)
parser.add_argument("--num_simulations", type=int, default=num_sim)

parser.add_argument("--resolution_x", type=int, default=96)
parser.add_argument("--resolution_y", type=int, default=128)
parser.add_argument("--buoyancy", type=float, default=-4e-3)
parser.add_argument("--bWidth", type=int, default=1)
parser.add_argument("--open_bound", type=bool, default=False)
parser.add_argument("--time_step", type=float, default=1.0)
parser.add_argument("--clamp_mode", type=int, default=1)

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

	p_list = np.linspace(args.min_src_x_pos, 
						   args.max_src_x_pos,
						   args.num_src_x_pos)
	pi_list = range(args.num_src_x_pos)

	# solver params
	res_x = args.resolution_x
	res_y = args.resolution_y
	gs = vec3(res_x, res_y, 1)
	buoyancy = vec3(0,args.buoyancy,0)

	solver = Solver(name='main', gridSize=gs, dim=2)
	solver.timestep = args.time_step
	
	flags = solver.create(FlagGrid)
	vel = solver.create(MACGrid)
	density = solver.create(RealGrid)
	pressure = solver.create(RealGrid)
	stream = solver.create(RealGrid)

	d_ = np.zeros([res_y,res_x], dtype=np.float32)
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
	d_range = [np.finfo(np.float).max, np.finfo(np.float).min]
	v_range = [np.finfo(np.float).max, np.finfo(np.float).min]
	p_range = [np.finfo(np.float).max, np.finfo(np.float).min]
	s_range = [np.finfo(np.float).max, np.finfo(np.float).min]
	for i, (p, pi) in enumerate(zip(p_list,pi_list)):
		start_time = time.time()
		
		flags.initDomain(boundaryWidth=args.bWidth)
		flags.fillGrid()
		if args.open_bound:
			setOpenBound(flags, args.bWidth,'xXyY', FlagOutflow|FlagEmpty)

		vel.clear()
		density.clear()
		pressure.clear()
		stream.clear()
		
		radius = gs.x*args.src_radius
		source = solver.create(Sphere, center=gs*vec3(p,args.src_y_pos,0.5), radius=radius)
		
		for t in range(args.num_frames):
			print('%s: simulating %d of %d (%d/%d)' % (datetime.now(), sim_id, num_total_sim, i+1, num_total_p))

			source.applyToGrid(grid=density, value=1)
				
			advectSemiLagrange(flags=flags, vel=vel, grid=density, order=2,
							   openBounds=args.open_bound, boundaryWidth=args.bWidth, clampMode=args.clamp_mode)
			advectSemiLagrange(flags=flags, vel=vel, grid=vel,     order=2,
							   openBounds=args.open_bound, boundaryWidth=args.bWidth, clampMode=args.clamp_mode)
			setWallBcs(flags=flags, vel=vel)
			addBuoyancy(density=density, vel=vel, gravity=buoyancy, flags=flags)
			solvePressure(flags=flags, vel=vel, pressure=pressure, cgMaxIterFac=10.0, cgAccuracy=0.0001)
			setWallBcs(flags=flags, vel=vel)
			getStreamfunction(flags=flags, vel=vel, grid=stream)
		
			copyGridToArrayReal(target=d_, source=density)
			copyGridToArrayReal(target=p_, source=pressure)
			copyGridToArrayReal(target=s_, source=stream)
			copyGridToArrayMAC(target=v_, _Source=vel)
			
			param_ = [p, t]

			d_range = [np.minimum(d_range[0], d_.min()),
					   np.maximum(d_range[1], d_.max())]
			v_range = [np.minimum(v_range[0], v_.min()),
					   np.maximum(v_range[1], v_.max())]
			p_range = [np.minimum(p_range[0], p_.min()),
					   np.maximum(p_range[1], p_.max())]
			s_range = [np.minimum(s_range[0], s_.min()),
					   np.maximum(s_range[1], s_.max())]

			v_file_path = os.path.join(args.log_dir, 'v', args.path_format % (pi, t))
			np.savez_compressed(v_file_path, 
								x=v_[:,:,:2],
								y=param_)

			s_file_path = os.path.join(args.log_dir, 's', args.path_format % (pi, t))
			np.savez_compressed(s_file_path, 
								x=np.expand_dims(s_, axis=-1),
								y=param_)

			# sim_file_path = os.path.join(args.log_dir, args.path_format % (pi, t))
			# np.savez_compressed(sim_file_path, 
			# 					d=np.expand_dims(d_, axis=-1),
			# 					p=np.expand_dims(p_, axis=-1),
			# 					s=np.expand_dims(s_, axis=-1),
			# 					v=v_[:,:,:2],
			# 					param=param_)

			solver.step()
			sim_id += 1
		
		gc.collect()
		duration = time.time() - start_time
		print('%s: done %d of %d (%.3f sec)' % (datetime.now(), sim_id, num_total_sim, duration))

	vrange_file = os.path.join(args.log_dir, 'v_range.txt')
	with open(vrange_file, 'w') as f:
		print('%s: velocity min %.3f max %.3f' % (datetime.now(), v_range[0], v_range[1]))
		f.write('%.3f\n' % v_range[0])
		f.write('%.3f' % v_range[1])

	srange_file = os.path.join(args.log_dir, 's_range.txt')
	with open(srange_file, 'w') as f:
		print('%s: stream min %.3f max %.3f' % (datetime.now(), s_range[0], s_range[1]))
		f.write('%.3f\n' % s_range[0])
		f.write('%.3f' % s_range[1])

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

	print('Done')


if __name__ == '__main__':
    main()