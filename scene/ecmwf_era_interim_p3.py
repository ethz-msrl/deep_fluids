import argparse
from datetime import datetime
import time
import os
from glob import glob
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

parser = argparse.ArgumentParser()

parser.add_argument("--log_dir", type=str, default='../data/ecmwf_era_interim')

parser.add_argument("--num_param", type=int, default=3)
parser.add_argument("--path_format", type=str, default='%d_%d_%d.npz')
parser.add_argument("--p0", type=str, default='year')
parser.add_argument("--p1", type=str, default='day')
parser.add_argument("--p2", type=str, default='time')

parser.add_argument("--num_year", type=int, default=5)
parser.add_argument("--min_year", type=float, default=2013)
parser.add_argument("--max_year", type=float, default=2017)
parser.add_argument("--num_day", type=int, default=365)
parser.add_argument("--min_day", type=float, default=1)
parser.add_argument("--max_day", type=float, default=365)
parser.add_argument("--num_time", type=int, default=9)
parser.add_argument("--min_time", type=float, default=0)
parser.add_argument("--max_time", type=float, default=24)
parser.add_argument("--num_simulations", type=int, default=16425)

parser.add_argument("--resolution_x", type=int, default=256) # 480
parser.add_argument("--resolution_y", type=int, default=128) # 240
parser.add_argument("--rescale", type=int, default=1)

args = parser.parse_args()

def main():
	# os.chdir(os.path.dirname(__file__))
	if not os.path.exists(args.log_dir):
		os.mkdir(args.log_dir)

	field_type = ['v']
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

	an_path = os.path.join(args.log_dir, 'uv-2013-2017-an.npy')
	an = np.load(an_path)[:,::-1,:,:] # flip
	v_range = [an.min(), an.max()]

	fc_paths = sorted(glob("{}/{}/*".format(args.log_dir, 'fc')))


	p1_ = np.linspace(args.min_year, 
					  args.max_year,
					  args.num_year)
	p2_ = np.linspace(args.min_day,
					  args.max_day,
					  args.num_day)
	p3_ = np.linspace(args.min_time,
					  args.max_time,
				  	  args.num_time)
	
	an_idx = 0
	fc_idx = 0
	for i in trange(args.num_year, desc='year'):
		p1 = p1_[i]
		for j in trange(args.num_day, desc='day'):
			p2 = p2_[j]
			if int(p2) == 366:
				an_idx += 2
				fc_idx += 8
				continue
			
			v_file_path = os.path.join(args.log_dir, 'v', args.path_format % (i, j, 0))
			v = an[an_idx,:-1,:,:]
			if args.rescale == 1: v = zoom(v, (args.resolution_y/240.0, args.resolution_x/480.0, 1))
			np.savez_compressed(v_file_path, 
								x=v,
								y=[p1, p2, p3_[0]])
		

			d = np.load(fc_paths[fc_idx+1])[::-1,:,:]
			v_range = [np.minimum(v_range[0], d.min()),
					   np.maximum(v_range[1], d.max())]
			v_file_path = os.path.join(args.log_dir, 'v', args.path_format % (i, j, 1))
			v = d[:-1,:,:]
			if args.rescale == 1: v = zoom(v, (args.resolution_y/240.0, args.resolution_x/480.0, 1))
			np.savez_compressed(v_file_path, 
								x=v,
								y=[p1, p2, p3_[1]])

			d = np.load(fc_paths[fc_idx+2])[::-1,:,:]
			v_range = [np.minimum(v_range[0], d.min()),
					   np.maximum(v_range[1], d.max())]
			v_file_path = os.path.join(args.log_dir, 'v', args.path_format % (i, j, 2))
			v = d[:-1,:,:]
			if args.rescale == 1: v = zoom(v, (args.resolution_y/240.0, args.resolution_x/480.0, 1))
			np.savez_compressed(v_file_path, 
								x=v,
								y=[p1, p2, p3_[2]])

			d = np.load(fc_paths[fc_idx+3])[::-1,:,:]
			v_range = [np.minimum(v_range[0], d.min()),
					   np.maximum(v_range[1], d.max())]
			v_file_path = os.path.join(args.log_dir, 'v', args.path_format % (i, j, 3))
			v = d[:-1,:,:]
			if args.rescale == 1: v = zoom(v, (args.resolution_y/240.0, args.resolution_x/480.0, 1))
			np.savez_compressed(v_file_path, 
								x=v,
								y=[p1, p2, p3_[3]])

			v_file_path = os.path.join(args.log_dir, 'v', args.path_format % (i, j, 4))
			v = an[an_idx+1,:-1,:,:]
			if args.rescale == 1: v = zoom(v, (args.resolution_y/240.0, args.resolution_x/480.0, 1))
			np.savez_compressed(v_file_path, 
								x=v,
								y=[p1, p2, p3_[4]])

			d = np.load(fc_paths[fc_idx+5])[::-1,:,:]
			v_range = [np.minimum(v_range[0], d.min()),
					   np.maximum(v_range[1], d.max())]
			v_file_path = os.path.join(args.log_dir, 'v', args.path_format % (i, j, 5))
			v = d[:-1,:,:]
			if args.rescale == 1: v = zoom(v, (args.resolution_y/240.0, args.resolution_x/480.0, 1))
			np.savez_compressed(v_file_path, 
								x=v,
								y=[p1, p2, p3_[5]])

			d = np.load(fc_paths[fc_idx+6])[::-1,:,:]
			v_range = [np.minimum(v_range[0], d.min()),
					   np.maximum(v_range[1], d.max())]
			v_file_path = os.path.join(args.log_dir, 'v', args.path_format % (i, j, 6))
			v = d[:-1,:,:]
			if args.rescale == 1: v = zoom(v, (args.resolution_y/240.0, args.resolution_x/480.0, 1))
			np.savez_compressed(v_file_path, 
								x=v,
								y=[p1, p2, p3_[6]])

			d = np.load(fc_paths[fc_idx+7])[::-1,:,:]
			v_range = [np.minimum(v_range[0], d.min()),
					   np.maximum(v_range[1], d.max())]
			v_file_path = os.path.join(args.log_dir, 'v', args.path_format % (i, j, 7))
			v = d[:-1,:,:]
			if args.rescale == 1: v = zoom(v, (args.resolution_y/240.0, args.resolution_x/480.0, 1))
			np.savez_compressed(v_file_path, 
								x=v,
								y=[p1, p2, p3_[7]])

			d = np.load(fc_paths[fc_idx+4])[::-1,:,:]
			v_range = [np.minimum(v_range[0], d.min()),
					   np.maximum(v_range[1], d.max())]
			v_file_path = os.path.join(args.log_dir, 'v', args.path_format % (i, j, 8))
			v = d[:-1,:,:]
			if args.rescale == 1: v = zoom(v, (args.resolution_y/240.0, args.resolution_x/480.0, 1))
			np.savez_compressed(v_file_path, 
								x=v,
								y=[p1, p2, p3_[8]])

			an_idx += 2
			fc_idx += 8

	vrange_file = os.path.join(args.log_dir, 'v_range.txt')
	with open(vrange_file, 'w') as f:
		print('%s: velocity min %.3f max %.3f' % (datetime.now(), v_range[0], v_range[1]))
		f.write('%.3f\n' % v_range[0])
		f.write('%.3f' % v_range[1])

	print('Done')


if __name__ == '__main__':
    main()