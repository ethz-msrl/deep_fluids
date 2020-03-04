import numpy as np
import time

class DivFreeRBF(object):

    def __init__(self, nodes, values, eps, kernel, normalize=False):
        """
        Params:
            * nodes (np.ndarray): the interpolation positions, should be Nx3
            * values (np.ndarray): should be Nx3
            * eps (float): the shape parameter of the kernel
            * kernel (str): the method used for the kernel (can be gaussian or multiquadric)
            * normalize (bool): if True, will normalize the positions to be between 0 and 1
        """
        self.Nd = nodes.shape[0]
        assert(values.shape[0] == self.Nd)
        assert(eps > 0)
        assert(kernel in ('gaussian', 'multiquadric'))

        self.eps = eps
        self.normalize = normalize

        if normalize:
            self.nmin = np.min(nodes, axis=0)
            self.nmax = np.max(nodes, axis=0)
            self.nodes = (nodes - self.nmin) / (self.nmax - self.nmin)
        else:
            self.nodes = nodes

        self.kernel = kernel

        self.eps2 = eps * eps

        self.coefficients = self.get_coefficients(values)

    def get_coefficients(self, values):
        """
        Gets the linear coefficients of the RBF 

        Params:
            * values (np.ndarray): should be Nx3

        Returns:
            np.ndarray of size Nx3 with the coefficients
        """
        [jj, ii] = np.meshgrid(np.arange(self.Nd), np.arange(self.Nd))

        if self.kernel == 'gaussian':
            K = np.exp(-self.eps * np.sum((self.nodes[jj, :] - self.nodes[ii, :])**2, axis=-1))

            psi_11 = (4 * self.eps - 4 * self.eps2 * ((self.nodes[jj,1] - self.nodes[ii,1])**2 + (self.nodes[jj,2] - self.nodes[ii, 2])**2)) * K
            psi_22 = (4 * self.eps - 4 * self.eps2 * ((self.nodes[jj,0] - self.nodes[ii,0])**2 + (self.nodes[jj,2] - self.nodes[ii, 2])**2)) * K
            psi_33 = (4 * self.eps - 4 * self.eps2 * ((self.nodes[jj,0] - self.nodes[ii,0])**2 + (self.nodes[jj,1] - self.nodes[ii, 1])**2)) * K

            psi_12 = 4 * self.eps2 * ((self.nodes[jj,0] - self.nodes[ii,0]) * 
                    (self.nodes[jj, 1] - self.nodes[ii, 1])) * K
            psi_13 = 4 * self.eps2 * ((self.nodes[jj,0] - self.nodes[ii,0]) * 
                    (self.nodes[jj, 2] - self.nodes[ii, 2])) * K
            psi_23 = 4 * self.eps2 * ((self.nodes[jj,1] - self.nodes[ii,1]) * 
                    (self.nodes[jj, 2] - self.nodes[ii, 2])) * K

        elif self.kernel == 'multiquadric':
            K = np.sqrt(1 + self.eps * np.sum((self.nodes[jj, :] - self.nodes[ii, :])**2, axis=-1))
            K3 = K ** 3

            psi_11 = (2 * self.eps / K - self.eps2 * ((self.nodes[jj,1] - self.nodes[ii,1])**2 + (self.nodes[jj,2] - self.nodes[ii, 2])**2)) / K3
            psi_22 = (2 * self.eps / K - self.eps2 * ((self.nodes[jj,0] - self.nodes[ii,0])**2 + (self.nodes[jj,2] - self.nodes[ii, 2])**2)) / K3
            psi_33 = (2 * self.eps / K - self.eps2 * ((self.nodes[jj,0] - self.nodes[ii,0])**2 + (self.nodes[jj,1] - self.nodes[ii, 1])**2)) / K3

            psi_12 = self.eps2 * ((self.nodes[jj,0] - self.nodes[ii,0]) * (self.nodes[jj, 1] - self.nodes[ii, 1])) / K3 
            psi_13 = self.eps2 * ((self.nodes[jj,0] - self.nodes[ii,0]) * (self.nodes[jj, 2] - self.nodes[ii, 2])) / K3
            psi_23 = self.eps2 * ((self.nodes[jj,1] - self.nodes[ii,1]) * (self.nodes[jj, 2] - self.nodes[ii, 2])) / K3

        A = np.concatenate((np.concatenate((psi_11, psi_12, psi_13)), 
            np.concatenate((psi_12, psi_22, psi_23)), 
            np.concatenate((psi_13, psi_23, psi_33))), axis=1)

        # because MATLAB uses column major and we're copying MATLAB code, it's simpler to use that 
        # form of indexing
        c = np.linalg.solve(A, values.flatten(order='F'))
        return c.reshape(self.Nd, 3, order='F')

    def evaluate_single(self, position):
        """ Evaluates the RBF at the given position

        Params:
            * position (np.ndarray) of length 3 with the position at which to evaluate the RBF
        Returns:
            np.array of length 3 with the evaluated RBF
        """
        assert(position.shape == (3,))

        if self.normalize:
            position = (position - self.nmin) / (self.nmax - self.nmin)

        if self.kernel == 'gaussian':
            K = np.exp(-self.eps * np.sum((position - self.nodes)**2, axis=-1))

            psi_11 = (4 * self.eps - 4 * self.eps2 * ((position[1] - self.nodes[:,1])**2 + (position[2] - self.nodes[:, 2])**2)) * K
            psi_22 = (4 * self.eps - 4 * self.eps2 * ((position[0] - self.nodes[:,0])**2 + (position[2] - self.nodes[:, 2])**2)) * K
            psi_33 = (4 * self.eps - 4 * self.eps2 * ((position[0] - self.nodes[:,0])**2 + (position[1] - self.nodes[:, 1])**2)) * K

            psi_12 = 4 * self.eps2 * ((position[0] - self.nodes[:,0]) * (position[1] - self.nodes[:,1])) * K 
            psi_13 = 4 * self.eps2 * ((position[0] - self.nodes[:,0]) * (position[2] - self.nodes[:,2])) * K
            psi_23 = 4 * self.eps2 * ((position[1] - self.nodes[:,1]) * (position[2] - self.nodes[:,2])) * K

        elif self.kernel == 'multiquadric':
            K = np.sqrt(1 + self.eps * np.sum((position - self.nodes)**2, axis=-1))
            K3 = K ** 3

            psi_11 = (2 * self.eps / K - self.eps2 * ((position[1] - self.nodes[:,1])**2 + (position[2] - self.nodes[:,2])**2)) / K3
            psi_22 = (2 * self.eps / K - self.eps2 * ((position[0] - self.nodes[:,0])**2 + (position[2] - self.nodes[:,2])**2)) / K3
            psi_33 = (2 * self.eps / K - self.eps2 * ((position[0] - self.nodes[:,0])**2 + (position[1] - self.nodes[:,1])**2)) / K3

            psi_12 = self.eps2 * ((position[0] - self.nodes[:,0]) * (position[1] - self.nodes[:,1])) / K3 
            psi_13 = self.eps2 * ((position[0] - self.nodes[:,0]) * (position[2] - self.nodes[:,2])) / K3
            psi_23 = self.eps2 * ((position[1] - self.nodes[:,1]) * (position[2] - self.nodes[:,2])) / K3

        Cv = self.coefficients.flatten(order='F') 
        bx = np.dot(np.concatenate((psi_11, psi_12, psi_13)), Cv)
        by = np.dot(np.concatenate((psi_12, psi_22, psi_23)), Cv)
        bz = np.dot(np.concatenate((psi_13, psi_23, psi_33)), Cv)
        return np.array([bx, by, bz])

    def evaluate(self, positions):
        """ Evaluates the RBF at an array of positions

        Params:
            * positions (np.ndarray) of length Nex3 with the Ne positions at which to evaluate the RBF
        Returns:
            np.array of length Nex3 with the evaluated RBF
        """
        Ne = positions.shape[0]
        assert(positions.shape[1] == 3)

        if self.normalize:
            positions = (positions - self.nmin) / (self.nmax - self.nmin)

        [jj, ii] = np.meshgrid(np.arange(Ne), np.arange(self.Nd), indexing='ij')

        if self.kernel == 'gaussian':
            K = np.exp(-self.eps * np.sum((positions[jj,:] - self.nodes[ii,:])**2, axis=-1))

            psi_11 = (4 * self.eps - 4 * self.eps2 * ((positions[jj,1] - self.nodes[ii,1])**2 + 
                (positions[jj,2] - self.nodes[ii,2])**2)) * K
            psi_22 = (4 * self.eps - 4 * self.eps2 * ((positions[jj,0] - self.nodes[ii,0])**2 + 
                (positions[jj,2] - self.nodes[ii,2])**2)) * K
            psi_33 = (4 * self.eps - 4 * self.eps2 * ((positions[jj,0] - self.nodes[ii,0])**2 + 
                (positions[jj,1] - self.nodes[ii,1])**2)) * K

            psi_12 = 4 * self.eps2 * ((positions[jj,0] - self.nodes[ii,0]) * (positions[jj,1] - self.nodes[ii,1])) * K 
            psi_13 = 4 * self.eps2 * ((positions[jj,0] - self.nodes[ii,0]) * (positions[jj,2] - self.nodes[ii,2])) * K
            psi_23 = 4 * self.eps2 * ((positions[jj,1] - self.nodes[ii,1]) * (positions[jj,2] - self.nodes[ii,2])) * K

        elif self.kernel == 'multiquadric':
            K = np.sqrt(1 + self.eps * np.sum((positions[jj,:] - self.nodes[ii,:])**2, axis=-1))
            K3 = K ** 3

            psi_11 = (2 * self.eps / K - self.eps2 * ((positions[jj,1] - self.nodes[ii,1])**2 + 
                (positions[jj,2] - self.nodes[ii,2])**2)) / K3
            psi_22 = (2 * self.eps / K - self.eps2 * ((positions[jj,0] - self.nodes[ii,0])**2 + 
                (positions[jj,2] - self.nodes[ii,2])**2)) / K3
            psi_33 = (2 * self.eps / K - self.eps2 * ((positions[jj,0] - self.nodes[ii,0])**2 + 
                (positions[jj,1] - self.nodes[ii,1])**2)) / K3

            psi_12 = self.eps2 * ((positions[jj,0] - self.nodes[ii,0]) * (positions[jj,1] - self.nodes[ii,1])) / K3 
            psi_13 = self.eps2 * ((positions[jj,0] - self.nodes[ii,0]) * (positions[jj,2] - self.nodes[ii,2])) / K3
            psi_23 = self.eps2 * ((positions[jj,1] - self.nodes[ii,1]) * (positions[jj,2] - self.nodes[ii,2])) / K3

        A = np.array([[psi_11, psi_12, psi_13], [psi_12, psi_22, psi_23], [psi_13, psi_23, psi_33]])
        return np.tensordot(A, self.coefficients, axes=((3,1), (0,1))).T
