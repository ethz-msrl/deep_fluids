import argparse
import numpy as np
import os
from tqdm import trange
from mpem import ElectromagnetCalibration
from mag_utils import grad5_to_grad33
import h5py

def get_positions(resolution, 
        x_min=-0.1, x_max=0.1,
        y_min=-0.1, y_max=0.1,
        z_min=-0.1, z_max=0.1):
    xv = np.linspace(x_min, x_max, resolution)
    yv = np.linspace(y_min, y_max, resolution)
    zv = np.linspace(z_min, z_max, resolution)

    zg, yg, xg = np.meshgrid(zv, yv, xv, indexing='ij')

    return xg, yg, zg

def divergence3(x, STEP=1.):
        dudx = (x[:-1,:-1,1:,0,:] - x[:-1,:-1,:-1,0,:])/STEP
        dvdy = (x[:-1,1:,:-1,1,:] - x[:-1,:-1,:-1,1,:])/STEP
        dwdz = (x[1:,:-1,:-1,2,:] - x[:-1,:-1,:-1,2,:])/STEP
        return dudx + dvdy + dwdz

def release_list(l):
    """
    This frees list l from memory immediately
    """

    del l[:]
    del l

if __name__ == '__main__':
    cal = ElectromagnetCalibration('/home/samuelch/tesla_ws/src/mag_control/mpem/cal/C_Mag_Calibration_06-25-2015.yaml')
    currents = np.loadtxt('/home/samuelch/datasets/cmag_calibration/currents_3787.csv')

    Nc = currents.shape[0]
    print('number of currents: %d' % Nc)

    parser = argparse.ArgumentParser(description='Generate some data using the MPEM framework that can be used \
    to train deep-fluids')

    parser.add_argument('--resolution', '-r', type=int, default=16)
    parser.add_argument('--train_ratio', type=float, default=0.9)
    parser.add_argument('--print_div', '-d',  action='store_true', help='print divergence info')
    parser.add_argument('--noise_std', '-g', type=float, help='add normally distribued \
            noise to the field data with the following standard deviation in T')
    parser.add_argument('--no_save', '-n', action='store_true', help='do not save data')
    parser.add_argument('--use_hdf5', action='store_true')
    parser.add_argument('--gradients', action='store_true', help='compute gradients instead of fields')
    parser.add_argument('dataset_dir', type=str)
    args = parser.parse_args()

    # We assume that the step size must be the same in all directions
    x_min = -0.1; x_max = 0.1
    y_min = -0.1; y_max = 0.1
    z_min = -0.1; z_max = 0.1

    step = (x_max - x_min) / (args.resolution - 1)

    Np = args.resolution ** 3
    xg, yg, zg = get_positions(args.resolution)
    xd = xg.ravel(); yd = yg.ravel(); zd = zg.ravel()

    n_train = int(args.train_ratio * Nc)
    n_test = Nc - n_train

    b_min = 0; b_max = np.inf

    print('Calculating actuation matrices')
    field_act_mats = [] 
    grad_act_mats = []
    if args.gradients:
        # calculating both field and gradient actuation matrices
        for i in trange(Np):
            act_mat = cal.fieldAndGradientCurrentJacobian(np.array([xd[i], yd[i], zd[i]]))
            field_act_mats.append(act_mat[0:3,:])
            grad_act_mats.append(act_mat[3:,:])
    else:
        for i in trange(Np):
            field_act_mats.append(cal.fieldCurrentJacobian(np.array([xd[i], yd[i], zd[i]])))

    #field_act_matx = np.array(field_act_mats)
    field_act_matx = np.reshape(field_act_mats, (args.resolution, args.resolution, args.resolution, 3, 8)) 
    release_list(field_act_mats)

    fields = field_act_matx.dot(currents.T)

    # freeing actuation matrices
    field_act_matx = None

    if args.noise_std:
        print('Adding noise')
        fields += args.noise_std * np.random.standard_normal(fields.shape)

    b_min = np.min(fields)
    b_max = np.max(fields)

    if args.gradients:
        #grad_act_matx = np.array(grad_act_mats)
        grad_act_matx = np.reshape(grad_act_mats, (args.resolution, args.resolution, args.resolution, 5, 8))
        release_list(grad_act_mats)

        gradient5s = grad_act_matx.dot(currents.T)

        gradient5s = gradient5s.transpose((4,0,1,2,3))
        
        # freeing actuation matrices
        grad_act_matx = None

        gradients = grad5_to_grad33(np.reshape(gradient5s, (-1, 5)))
        gradient5s = None
        gradients = gradients.reshape((Nc, args.resolution, args.resolution, args.resolution, 3, 3))

    if args.print_div:
        div3 = divergence3(fields, step)
        print('divergence mean: %f, max: %f, min: %f' % (np.mean(div3), np.max(div3), np.min(div3)))

    if not args.no_save:
        dataset_dir = args.dataset_dir
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        train_dir  = os.path.join(dataset_dir, 'v')

        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        else:
            filelist = [f for f in os.listdir(train_dir) if f.endswith('.npz')]
            for f in filelist:
                os.remove(os.path.join(train_dir, f))

        test_dir = os.path.join(dataset_dir, 'test')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        else:
            filelist = [f for f in os.listdir(test_dir) if f.endswith('.npz')]
            for f in filelist:
                os.remove(os.path.join(test_dir, f))

        if args.use_hdf5:
            with h5py.File(os.path.join(dataset_dir, 'positions.h5'), 'w') as f:
                f.create_dataset('xg', data=xg)
                f.create_dataset('yg', data=yg)
                f.create_dataset('zg', data=zg)
        else:
            np.savez(os.path.join(dataset_dir, 'positions.npz'), xg=xg, yg=yg, zg=zg) 

        print('Writing out field data')
        for i in trange(Nc):
            x = fields[:,:,:,:,i]

            if i < n_train:
                filepath = os.path.join(train_dir, '%04d' % i)
            else:
                filepath = os.path.join(test_dir, '%04d' % i)

            if args.use_hdf5:
                filepath += '.h5'
                with h5py.File(filepath, 'w') as hf:
                    hf.create_dataset('fields', data=x)
                    if args.gradients:
                        # indices is current, x, y, z, 3, 3
                        g = gradients[i,:,:,:,:,:]
                        hf.create_dataset('gradients', data=g)
            else:
                filepath += '.npz'
                np.savez_compressed(filepath, x=x, y=currents[i,:])

        args_file = os.path.join(dataset_dir, 'args.txt')
        with open(args_file, 'w') as f:
            f.write('num_param: 8\n')
            f.write('min_c: -35\n')
            f.write('max_c: 35\n')
            f.write('min_b: %f\n' % b_min)
            f.write('max_b: %f\n' % b_max)
            f.write('res: %d\n' % args.resolution)
            f.write('sampling: 1\n')
