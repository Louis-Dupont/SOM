import numpy as np
from sklearn.decomposition import PCA


class Node():
    """
        Node of the lattice
            position = The position inside the lattice
            weight = The position within the input space
            weight_buffer = The value that will be added to the Node weight in the next update
            nb_buffer = number of "buffers"
    """
    def __init__(self, lattice, *args):
        self.lattice = lattice
        self.position = np.array(args)
        self.weight   = np.random.rand(*lattice.input_shape)
        
        self.weight_buffer = np.zeros(*lattice.input_shape)
        self.nb_buffer = 0
        
    def get_weight(self):
        return self.weight
    def get_position(self):
        return self.position
    
    def set_weight(self, weight):
        self.weight = weight
    
    
    def distance_to(self, vector):
        return self.lattice.distance_function(vector - self.weight)
    
    def add_weight_buffer(self, delta):
        self.weight_buffer += delta
        self.nb_buffer += 1
        
    def update_weight(self, average_buffer = False):
        if average_buffer:
            self.weight  += self.weight_buffer/self.nb_buffer
        else:
            self.weight  += self.weight_buffer            
        self.weight_buffer = np.zeros(*self.lattice.input_shape)
        self.nb_buffer = 0
        
    def __getitem__(self, i):
        return self.position[i]
    def __setitem__(self, i, value):
        self.position[i] = value
    
    def __add__(self, other):
        try:
            if isinstance(other, Node):
                return other.weight + self.weight
            else:
                return other + self.weight
        except TypeError:
            raise TypeError(f"type {type(self.weight)} and type {type(other)} can not be + ")

    
    def __sub__(self, other):
        try:
            if isinstance(other, Node):
                return other.weight - self.weight
            else:
                return other - self.weight
        except TypeError:
            raise TypeError(f"type {type(self.weight)} and type {type(other)} can not be - ")
    
    def __mul__(self, other):
        try:
            if isinstance(other, Node):
                return other.weight * self.weight
            else:
                return other * self.weight
        except TypeError:
            raise TypeError(f"type {type(self.weight)} and type {type(other)} can not be multiplied")
         
    def __radd__(self, other):
        raise TypeError("please try to change the order (a+b)-> (b+a)")  
    def __rsub__(self, other):
        raise TypeError("please try to change the order (a-b)-> -(b-a)") 
    def __rmult__(self, other):
        raise TypeError("please try to change the order (a*b)-> (b*a)")
        
    def __iadd__(self, other):
        self.weight = self + other
        return self
    def __isub__(self, other):
        self.weight = self - other
        return self    
    def __imult__(self, other):
        self.weight = self * other
        return self
        
            
class Lattice():
    """
        n-dimensional Lattice composed of "Node" Objects
            lattice = np.ndarray containing every Node
            distance_function = usually euclidean distance
    
    """
    def __init__(self, distance_function = np.linalg.norm):
        self.input_shape = None
        self.lattice = None
        self.distance_function = distance_function
        self.set_update_function(epsilon_0 = 0.8, 
                                 sigma_0   = 20, 
                                 t_max     = 100)
             
    def __getitem__(self, position):
        return self.lattice[position]   
    def __setitem__(self, position, value):
        self.lattice[position] = value
    

    def node_distance(self, node_1, node_2):
        return self.distance_function(node_1.position-node_2.position)
    
    def set_update_function(self, epsilon_0, sigma_0, t_max):
        self.update_function = Update(self.node_distance, epsilon_0, sigma_0, t_max)
        
      
    def get_node_weight(self, position):
        return self[position].get_weight()  
    def get_weight(self):
        f = np.vectorize(lambda node: node.get_weight(), signature='()->(n)')
        return f(self.lattice)
    def get_position(self):
        f = np.vectorize(lambda node: node.get_position(), signature='()->(n)')
        return f(self.lattice)
    
    
    
    def relative_distance(self, ref_vector):
        f = np.vectorize(lambda node: node.distance_to(ref_vector))
        return f(self.lattice)
    
    def update_nodes(self):
        for node in self:
            node.update_weight()
        
    def find_closest_id(self, ref_vector):
        distance_matrix = self.relative_distance(ref_vector)
        return np.unravel_index(distance_matrix.argmin(), distance_matrix.shape)
    
    def find_closest(self, ref_vector):
        return self.lattice[self.find_closest_id(ref_vector)]
        
        

    
    def run(self, input_vector, t, batch_update):

        best_matching_node = self.find_closest(input_vector)
        for node in self:  
            update_value = self.update_function.value( reference_vector   = input_vector, 
                                                       best_matching_node = best_matching_node,
                                                       current_node       = node,
                                                       t                  = t) 
            
            node.add_weight_buffer(update_value*(node-input_vector))
            if not(batch_update):
                node.update_weight()

        
    def train(self, batch, batch_update=False):
        for t, input_vector in enumerate(batch):
             self.run(input_vector, t, batch_update)
        self.get_position() 
        if batch_update:
            self.update_nodes() 
    
    def __iter__(self):
        pass
    
    
    

    
class Lattice_1D(Lattice):
    def __init__(self, input_shape, lattice_dim):
        super().__init__()
        self.input_shape = input_shape
        self.lattice_dim = lattice_dim
        self.lattice = np.array([Node(self, i) for i in range(lattice_dim[0])])
                
    def __iter__(self):
        for i in range(self.lattice_dim[0]):
            yield self.lattice[i]
            
    def init_weight_pca(self, batch):
        pca = PCA(n_components=1)
        pca.fit(batch)  
        pca_batch = pca.fit_transform(batch)


        pca_sample = np.linspace(np.min(pca_batch, axis = 0), np.max(pca_batch, axis = 0), self.lattice_dim)

        pca_sample_reverse = pca.inverse_transform(pca_sample)
        for node, vector in zip(self, pca_sample_reverse):
            node.set_weight(vector)
            
            
class Lattice_2D(Lattice):
    def __init__(self, input_shape, lattice_dim):
        super().__init__()
        self.input_shape = input_shape
        self.lattice_dim = lattice_dim
        self.lattice = np.array([[Node(self, i, j) for j in range(lattice_dim[1])] 
                                                   for i in range(lattice_dim[0])])
        
    
    def __iter__(self):
        for i in range(self.lattice_dim[0]):
            for j in range(self.lattice_dim[1]):
                yield self.lattice[i][j]
                
            
    def init_weight_pca(self, batch):
        pca = PCA(n_components=2)
        pca.fit(batch)  
        pca_batch = pca.fit_transform(batch)


        x_axis = np.linspace(np.min(pca_batch, axis = 0)[0], np.max(pca_batch, axis = 0)[0], self.lattice_dim[0])
        y_axis = np.linspace(np.min(pca_batch, axis = 0)[1], np.max(pca_batch, axis = 0)[1], self.lattice_dim[1])

        xv, yv = np.meshgrid(x_axis, y_axis)
        x_sample = xv.reshape([-1])
        y_sample = yv.reshape([-1])
        xy_sample = np.concatenate([[x_sample, y_sample]], axis=0).swapaxes(1,0)
        pca_sample_reverse = pca.inverse_transform(xy_sample)
        
        for node, vector in zip(self, pca_sample_reverse):
            node.set_weight(vector)

                
# f(x) = x_0 * exp(-x/x_max)
class DecreasingExp():
    def __init__(self, x_0, x_max):
        self.x_0 = x_0
        self.x_max = x_max
    def value(self, x):
        return self.x_0*np.exp(-x/self.x_max)

# class Normal():
#     def __init__(self, node_distance, sigma):
#         self.node_distance = node_distance
#         self.sigma    = sigma
#     def value(self, node_i, node_j, t):
#         d_ij      = self.node_distance(node_i, node_j)
#         sigma_t   = self.sigma.value(t)
#         return np.exp(-d_ij**2/(2*sigma_t**2))

class Update():
    def __init__(self, node_distance, epsilon_0, sigma_0, t_max):
        self.node_distance = node_distance

        self.epsilon   = DecreasingExp(epsilon_0, t_max)
        self.sigma     = DecreasingExp(sigma_0,   t_max)
#         self.etha      = Normal(self.node_distance, self.sigma)

    def value(self, reference_vector, best_matching_node, current_node, t):
        epsilon_t = self.epsilon.value(t)
        sigma_t   = self.sigma.value(t)
        d_ij      = self.node_distance(best_matching_node, current_node)
#         etha_t_ij = self.etha.value(best_matching_node, current_node, t)
        
        return epsilon_t * np.exp(-d_ij**2/(2*sigma_t**2))
