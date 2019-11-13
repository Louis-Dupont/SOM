import seaborn as sns; 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_list_distance_matrix(lattice, list_ref_vector):
    max_width = 15
    number_vectors = list_ref_vector.shape[0]
    height = max_width // number_vectors
    fig, axs = plt.subplots(ncols=number_vectors, figsize=(max_width, height))
    for i, vector in enumerate(list_ref_vector):
        sns.heatmap(lattice.relative_distance(vector), vmin=0, ax=axs[i])

        
def plot2D(lattice=None, ref_values=None):

    if lattice is not None:
        lattice_dim = lattice.lattice_dim.shape[0]
        if lattice_dim == 2:
            lattice_x = lattice.get_weight()[:, :, 0].reshape([-1])
            lattice_y = lattice.get_weight()[:, :, 1].reshape([-1])
            plt.plot(lattice_x, lattice_y)
        elif lattice_dim == 1:
            lattice_x = lattice.get_weight()[:, 0].reshape([-1])
            lattice_y = lattice.get_weight()[:, 1].reshape([-1])
            plt.plot(lattice_x, lattice_y)
    
    
    if ref_values is not None:
        X, Y = ref_values
        plt.scatter(X, Y)
        
    

def plot3D(lattice=None, ref_values=None):

    fig = plt.figure()
    ax = Axes3D(fig)
    if lattice is not None:
        lattice_dim = lattice.lattice_dim.shape[0]
        if lattice_dim == 2:
            lattice_x = lattice.get_weight()[:, :, 0].reshape([-1])
            lattice_y = lattice.get_weight()[:, :, 1].reshape([-1])
            lattice_z = lattice.get_weight()[:, :, 2].reshape([-1])
            ax.scatter(lattice_x, lattice_y, lattice_z)
        elif lattice_dim == 1:
            lattice_x = lattice.get_weight()[:, 0].reshape([-1])
            lattice_y = lattice.get_weight()[:, 1].reshape([-1])
            lattice_z = lattice.get_weight()[:, 2].reshape([-1])
            ax.plot(lattice_x, lattice_y, lattice_z)
    
    
    if ref_values is not None:
        X, Y, Z = ref_values
        ax.scatter(X, Y, Z)
        
    plt.show()