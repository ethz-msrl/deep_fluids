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

def str2bool(v):
    return v.lower() in ('true', '1')

parser = argparse.ArgumentParser()

parser.add_argument("--log_dir", type=str, default='data/smoke_pos21_size5_f200')

parser.add_argument("--num_param", type=int, default=3)
parser.add_argument("--path_format", type=str, default='%d_%d_%d.npz')
parser.add_argument("--p0", type=str, default='src_x_pos')
parser.add_argument("--p1", type=str, default='src_radius')
parser.add_argument("--p2", type=str, default='frames')

parser.add_argument("--num_src_x_pos", type=int, default=10)
parser.add_argument("--min_src_x_pos", type=float, default=0.2)
parser.add_argument("--max_src_x_pos", type=float, default=0.8)
parser.add_argument("--src_y_pos", type=float, default=0.1)
parser.add_argument("--num_src_radius", type=int, default=10)
parser.add_argument("--min_src_radius", type=float, default=0.04)
parser.add_argument("--max_src_radius", type=float, default=0.12)
parser.add_argument("--num_frames", type=int, default=200)
parser.add_argument("--min_frames", type=int, default=0)
parser.add_argument("--max_frames", type=int, default=199)
parser.add_argument("--num_simulations", type=int, default=21000)

parser.add_argument("--resolution_x", type=int, default=96)
parser.add_argument("--resolution_y", type=int, default=128)
parser.add_argument("--buoyancy", type=float, default=-4e-3)
parser.add_argument("--bWidth", type=int, default=1)
parser.add_argument("--open_bound", type=bool, default=False)
parser.add_argument("--time_step", type=float, default=0.5)
parser.add_argument("--clamp_mode", type=int, default=2)
parser.add_argument("--strength", type=float, default=0.1)

parser.add_argument('--is_test', type=str2bool, default=False)
parser.add_argument('--vpath', type=str, default='')

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

def test():
	filename = os.path.basename(args.vpath)[:-4]
	print(filename)

	# p1, p2 = filename.split('_')
	# p1, p2 = get_param(float(p1), float(p2))	
	# p1 = 0.48
	# p2 = 0.08
	p1, p2 = get_param(float(4), float(2))	
	title = 'd_' + filename
	img_dir = os.path.join(os.path.dirname(args.vpath), filename + '_')

	# solver params
	res_x = args.resolution_x
	res_y = args.resolution_y
	gs = vec3(res_x, res_y, 1)
	buoyancy = vec3(0,args.buoyancy,0)

	s = Solver(name='main', gridSize=gs, dim=2)
	s.timestep = args.time_step
	
	flags = s.create(FlagGrid)
	vel = s.create(MACGrid)
	density = s.create(RealGrid)
	pressure = s.create(RealGrid)


	flags = s.create(FlagGrid)
	vel = s.create(MACGrid)
	density = s.create(RealGrid)

	if (GUI):
		gui = Gui()
		gui.show(True)
		gui.nextVec3Display()
		gui.nextVec3Display()
		gui.nextVec3Display()
		# gui.pause()

	flags.initDomain(boundaryWidth=args.bWidth)
	flags.fillGrid()

	vel.clear()
	density.clear()

	# v_path = 'log/nn/velocity/smoke_mov500_f200_0424_174149_8_1K_diff/v.npz'
	# with np.load(v_path) as data:
	# 	v = data['v']
	# 	p = data['p']
	
	# v_path = 'log/nn/velocity/smoke_mov500_f200_0424_174149_8_1K_diff/v_.npz'
	# with np.load(v_path) as data:
	# 	v_gt = data['v_gt']

	# print(v.shape, v_gt.shape, p.shape)

	from PIL import Image
	d_ = np.zeros([args.resolution_y,args.resolution_x], dtype=np.float32)
	with np.load(args.vpath) as data:
		x = data['v']

	for i in range(args.num_frames):
		v = x[i,...]
		v = np.dstack((v[::-1,:,:],np.zeros([args.resolution_y, args.resolution_x, 1])))

		# v_ = np.dstack((v_gt[i,::-1,:,:],np.zeros([args.resolution_y, args.resolution_x, 1])))
		# v_ = np.dstack((v[i,::-1,:,:],np.zeros([args.resolution_y, args.resolution_x, 1])))
		copyArrayToGridMAC(target=vel, source=v)

		source = s.create(Sphere, center=gs*vec3(p1,args.src_y_pos,0.5), radius=gs.x*p2)
		source.applyToGrid(grid=density, value=1)			
		advectSemiLagrange(flags=flags, vel=vel, grid=density, order=2,
							openBounds=True, boundaryWidth=args.bWidth, clampMode=args.clamp_mode)

		copyGridToArrayReal(target=d_, source=density)
		# img = np.expand_dims(d_[::-1,:] * 255.0, axis=-1)
		img = np.uint8((1-d_[::-1,:])*255)

		img_path = os.path.join(img_dir, '%04d.png' % i)
		im = Image.fromarray(img)
		im.save(img_path)
		s.step()
		# gui.screenshot(os.path.join('log/nn/velocity/smoke_mov500_f200_0424_174149_8_1K_diff/scr_gt_','%04d.png' % i))

def main():
	if not os.path.exists(args.log_dir):
		os.mkdir(args.log_dir)

	field_type = ['v'] # , 's'] # 'p'
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
	buoyancy = vec3(0,args.buoyancy,0)

	solver = Solver(name='main', gridSize=gs, dim=2)
	solver.timestep = args.time_step
	
	flags = solver.create(FlagGrid)
	vel = solver.create(MACGrid)
	density = solver.create(RealGrid)
	pressure = solver.create(RealGrid)
	# stream = solver.create(RealGrid)

	v_ = np.zeros([res_y,res_x,3], dtype=np.float32)
	# d_ = np.zeros([res_y,res_x], dtype=np.float32)
	# p_ = np.zeros([res_y,res_x], dtype=np.float32)
	# s_ = np.zeros([res_y,res_x], dtype=np.float32)

	if (GUI):
		gui = Gui()
		gui.show(True)
		#gui.pause()

	print('start generation')
	sim_id = 0
	num_total_p = p_list.shape[0]
	num_total_sim = num_total_p * args.num_frames
	v_range = [np.finfo(np.float).max, np.finfo(np.float).min]
	# d_range = [np.finfo(np.float).max, np.finfo(np.float).min]
	# p_range = [np.finfo(np.float).max, np.finfo(np.float).min]
	# s_range = [np.finfo(np.float).max, np.finfo(np.float).min]
	for i, (p, pi) in enumerate(zip(p_list,pi_list)):
		start_time = time.time()
		
		flags.initDomain(boundaryWidth=args.bWidth)
		flags.fillGrid()
		if args.open_bound:
			setOpenBound(flags, args.bWidth,'xXyY', FlagOutflow|FlagEmpty)

		vel.clear()
		density.clear()
		pressure.clear()
		# stream.clear()
		
		radius = gs.x*p[1]
		source = solver.create(Sphere, center=gs*vec3(p[0],args.src_y_pos,0.5), radius=radius)
		
		for t in range(args.num_frames):
			print('%s: simulating %d of %d (%d/%d)' % (datetime.now(), sim_id, num_total_sim, i+1, num_total_p))

			source.applyToGrid(grid=density, value=1)
				
			advectSemiLagrange(flags=flags, vel=vel, grid=density, order=2,
							   openBounds=args.open_bound, boundaryWidth=args.bWidth, clampMode=args.clamp_mode)
			advectSemiLagrange(flags=flags, vel=vel, grid=vel,     order=2,
							   openBounds=args.open_bound, boundaryWidth=args.bWidth, clampMode=args.clamp_mode)

			if args.strength > 0:
				vorticityConfinement(vel=vel, flags=flags, strength=args.strength)

			setWallBcs(flags=flags, vel=vel)
			addBuoyancy(density=density, vel=vel, gravity=buoyancy, flags=flags)
			solvePressure(flags=flags, vel=vel, pressure=pressure, cgMaxIterFac=10.0, cgAccuracy=0.0001)
			setWallBcs(flags=flags, vel=vel)
			# getStreamfunction(flags=flags, vel=vel, grid=stream)
		
			copyGridToArrayMAC(target=v_, _Source=vel)
			# copyGridToArrayReal(target=d_, source=density)
			# copyGridToArrayReal(target=p_, source=pressure)
			# copyGridToArrayReal(target=s_, source=stream)
			
			param_ = [p[0], p[1], t]

			v_range = [np.minimum(v_range[0], v_.min()),
					   np.maximum(v_range[1], v_.max())]
			# d_range = [np.minimum(d_range[0], d_.min()),
			# 		   np.maximum(d_range[1], d_.max())]
			# p_range = [np.minimum(p_range[0], p_.min()),
			# 		   np.maximum(p_range[1], p_.max())]
			# s_range = [np.minimum(s_range[0], s_.min()),
			# 		   np.maximum(s_range[1], s_.max())]

			v_file_path = os.path.join(args.log_dir, 'v', args.path_format % (pi[0], pi[1], t))
			np.savez_compressed(v_file_path, 
								x=v_[:,:,:2],
								y=param_)

			# s_file_path = os.path.join(args.log_dir, 's', args.path_format % (pi[0], pi[1], t))
			# np.savez_compressed(s_file_path, 
			# 					x=np.expand_dims(s_, axis=-1),
			# 					y=param_)

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

	# srange_file = os.path.join(args.log_dir, 's_range.txt')
	# with open(srange_file, 'w') as f:
	# 	print('%s: stream min %.3f max %.3f' % (datetime.now(), s_range[0], s_range[1]))
	# 	f.write('%.3f\n' % s_range[0])
	# 	f.write('%.3f' % s_range[1])

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
	if args.is_test:
		test()
	else:
		main()