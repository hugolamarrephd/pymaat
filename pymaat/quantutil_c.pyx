import numpy as np
from cython.parallel import prange, parallel

cimport numpy as np
cimport cython
cimport openmp
from libc.math cimport HUGE_VAL
from libc.math cimport fabs, fmin, ceil
from libc.stdlib cimport malloc, free
from cpython.mem cimport PyMem_Malloc, PyMem_Free

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def inplace_clvq(
        double[:,::1] simulations, double[:,::1] quantizer, double[:] weight,
        double g0, double a, double b, int N, double cumulative_distortion):
    """
    In-place competitive learning vector quantization (CLVQ) algorithm
    Also, returns distortion companion estimate
    """
    cdef:
        unsigned int size
        unsigned int n, ndim
        unsigned int t, nsim
        unsigned int idx
        double g
        double distortion
        double *xi
    size = quantizer.shape[0]
    ndim = quantizer.shape[1]
    nsim = simulations.shape[0]
    for t in range(nsim):
        xi = &simulations[t,0]
        idx = _naive_nearest(
                quantizer, xi, &weight[0], size, ndim, &distortion)
        # Perform CLVQ step
        g = g0*a/(a+g0*b*<double>N)
        cumulative_distortion += distortion
        for n in range(ndim):
            quantizer[idx,n] -= g*(quantizer[idx,n]-xi[n])
        N += 1
    return N, cumulative_distortion

@cython.boundscheck(False)
@cython.initializedcheck(False)
def lloyd1(
        double[:,:] simulations, double[:,:] quantizer, double[:] weight,
        unsigned long[:] count, double[:,:] cumul):
    cdef:
        unsigned int t, nsim
        unsigned int n, ndim
        unsigned int[:] idx
        KDTree tree
    ndim = quantizer.shape[1]
    nsim = simulations.shape[0]
    idx = KDTree(quantizer).nearest(simulations, weight)
    for t in range(nsim):
        count[idx[t]] += 1
        for n in range(ndim):
            cumul[idx[t],n] += simulations[t,n]


#########################
# Special Quantizations #
#########################

@cython.boundscheck(False)
@cython.initializedcheck(False)
def digitize_1d(double[:] voronoi, double[:] values):
    """
    Return indices of values in voronoi grid
        assum voronoi is strictly increasing
    """
    cdef:
        unsigned int i, size
        unsigned int t, nval
        np.int_t[:] idx

    size = voronoi.size-1
    nval = values.size

    idx = np.zeros((nval,), dtype=np.int_)

    for t in range(nval):
        for i in range(size):
            if values[t]<voronoi[i+1]:
                idx[t] = i
                break

    return idx

@cython.boundscheck(False)
@cython.initializedcheck(False)
def conditional_digitize_2d(
        double[:] voronoi1, double[:,:] voronoi2, double[:,:] values):
    """
    Return indices of values in voronoi grid
        assum voronoi is strictly increasing
    """
    cdef:
        unsigned int i, size1
        unsigned int j, size2
        unsigned int t, nval
        np.int_t[:] idx1, idx2

    size1 = voronoi1.shape[0]-1
    size2 = voronoi2.shape[1]-1
    nval = values.shape[0]

    idx1 = np.zeros((nval,), dtype=np.int_)
    idx2 = np.zeros((nval,), dtype=np.int_)

    for t in range(nval):
        for i in range(size1):
            if values[t,0]<voronoi1[i+1]:
                idx1[t] = i
                for j in range(size2):
                    if values[t,1]<voronoi2[i,j+1]:
                        idx2[t] = j
                        break
                break

    return idx1, idx2


#####################
# Nearest-Neighbor #
#####################

def naive_nearest(
        double[:,::1] quantizer, double[:,::1] at, double[:] weight):
    """
    Return indices of closest quantizer for at, using euclidean distance
    """
    cdef:
        int t, T=at.shape[0]
        unsigned int size=quantizer.shape[0], ndim=quantizer.shape[1]
        np.ndarray[double, ndim=2, mode='c'] rows
        np.ndarray[unsigned int, ndim=1, mode='c'] result
        double _
    result = np.empty((T,), dtype=np.uint32)
    for t in prange(T, nogil=True, schedule='guided'):
        result[t] = _naive_nearest(
                quantizer, &at[t,0], &weight[0], size, ndim, &_)
    return result


@cython.boundscheck(False)
@cython.initializedcheck(False)
cdef unsigned int _naive_nearest(
        double[:,::1] quantizer, double* at, double* weight,
        unsigned int size, unsigned int ndim, double *distance) nogil:
    cdef:
        unsigned int idx
        double tmp, d, min_d
        double *current
    idx = 0
    min_d = HUGE_VAL
    for i in range(size):
        current = &quantizer[i,0]
        d = 0.
        for n in range(ndim):
            tmp = (current[n] - at[n])*weight[n]
            d += tmp*tmp
        if d<min_d:
            idx = i
            min_d = d
    distance[0] = min_d
    return idx


cdef struct KDTreeNode:
    np.float_t *low
    np.float_t *high
    unsigned int low_idx, high_idx
    # Tree structure...
    unsigned int parent
    unsigned int left, right

cdef class KDTree:

    cdef:
        double[::1,:] values  # Fortran convention for fast column extract
        unsigned int size, ndim, nboxes
        unsigned int leafsize
        KDTreeNode* boxes
        # Internals
        unsigned int *_idx

    def __cinit__(self, double[:,:] values not None, leafsize=None):
        self.values = np.copy(np.asfortranarray(values))
        self.size = self.values.shape[0]
        self.ndim = self.values.shape[1]
        if leafsize is None:
            leafsize = self.size
        # Memory allocation...
        self._allocate_memory()
        # Initialize tree...
        self._init_root_box()
        # Build tree
        self._build(leafsize)

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    def nearest(self, double[:,:] at not None, double[:] weight not None):
        cdef:
            int t, T=at.shape[0]
            np.ndarray[double, ndim=2, mode='c'] rows
            np.ndarray[unsigned int, ndim=1, mode='c'] result
        at = np.ascontiguousarray(at)
        # TODO: check at and weight shape here
        result = np.empty((T,), dtype=np.uint32)
        for t in prange(T, nogil=True, schedule='guided'):
            result[t] = self._nearest(&at[t,0], &weight[0])
        return result

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cdef unsigned int _nearest(self, double *at, double *weight) nogil:
        cdef:
            unsigned int location
            unsigned int nearest
            unsigned int node
            unsigned int task_id
            double distance
            unsigned int *_query_nodes
            KDTreeNode box
            unsigned int i, n
            double d, tmp
            unsigned int dim = 0
            unsigned int right, left, is_left
            double low, high
        distance = HUGE_VAL
        # Step 1. Find nearest in box
        # In which box?
        location = 0
        dim = 0
        while self.boxes[location].left:  # has siblings?
            left = self.boxes[location].left
            right = self.boxes[location].right
            is_left = at[dim] <= self.boxes[left].high[dim]
            location = left if is_left else right
            dim = (dim+1) % self.ndim
        # Nearest in box?
        box = self.boxes[location]
        for i in range(box.low_idx, box.high_idx+1):
            d = 0.
            for n in range(self.ndim):
                tmp = (
                        self.values[self._idx[i],n] - at[n]
                        ) * weight[n]
                d += tmp*tmp
            if d < distance:
                nearest = self._idx[i]
                distance = d
        # Step 2. Traverse tree
        # Initialize tasks
        _query_nodes = <unsigned int*> malloc(
                50 * sizeof(unsigned int))
        _query_nodes[1] = 0
        task_id = 1
        while task_id:
            node = _query_nodes[task_id]
            task_id -= 1
            box = self.boxes[node]
            # Distance from box?
            d = 0.
            for n in range(self.ndim):
                low = box.low[n]
                high = box.high[n]
                if at[n] < low:
                    tmp = (at[n] - low) * weight[n]
                    d += tmp*tmp
                if at[n] > high:
                    tmp = (at[n] - high) * weight[n]
                    d += tmp*tmp
            if d < distance:
                # Found closer box
                if box.left:  # has siblings
                    # Explore siblings...
                    task_id += 1
                    _query_nodes[task_id] = box.left
                    task_id += 1
                    _query_nodes[task_id] = box.right
                else: # is a leaf
                    # Nearest in box...
                    for i in range(box.low_idx, box.high_idx+1):
                        d = 0.
                        for n in range(self.ndim):
                            tmp = (
                                    self.values[self._idx[i],n] - at[n]
                                    ) * weight[n]
                            d += tmp*tmp
                        if d < distance:
                            nearest = self._idx[i]
                            distance = d
        free(_query_nodes)
        return nearest

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)  # For modulo
    cdef _build(self, unsigned int leafsize):
        cdef:
            unsigned int box=0,
            unsigned int parent, dim
            unsigned int p_idx
            unsigned int at
            unsigned int left, mid, right
            double value
            unsigned int next_dim
            unsigned int nleaf
            unsigned int task_id
            unsigned int *task_nodes
            unsigned int *task_dims
            unsigned int i
            unsigned int location, is_left
        # Initialize result
        for i in range(self.size):
            self._idx[i] = i
        # Initialize tasks
        task_nodes = <unsigned int*> malloc(50*sizeof(unsigned int))
        task_dims = <unsigned int*> malloc(50*sizeof(unsigned int))
        if self.nboxes==1: # Handle empty tree case...
            task_id = 0
        else:
            task_nodes[1] = 0
            task_dims[1] = 0
            task_id = 1
        nleaf = 1
        while task_id:
            # Kill tree if more than leafsize
            nleaf += 1
            if nleaf > leafsize:
                break
            # Retrieve current task (parent and dimension)
            parent = task_nodes[task_id]
            dim = task_dims[task_id]
            task_id -= 1
            # Partition
            left = self.boxes[parent].low_idx
            right = self.boxes[parent].high_idx
            at = (right-left)>>1
            mid = left + at
            p_idx = _partition_idx(  # split_idx points to the
                    at,  # median value between low-high of parent node
                    &self._idx[left],
                    &self.values[0,dim],  # values is Fortran array
                    right+1-left
                    )
            value = self.values[p_idx,dim]
            # Compute next dimension
            next_dim = (dim+1) % self.ndim
            # Make left box
            box += 1
            self._fork_at(box, parent, left, mid)
            self.boxes[parent].left = box
            self.boxes[box].high[dim] = value
            if mid-left > 1:
                # At least 3 points remaining in left box?
                # ...request a left-split
                task_id += 1
                task_nodes[task_id] = box
                task_dims[task_id] = next_dim
            # Make right box
            box += 1
            self._fork_at(box, parent, mid+1, right)
            self.boxes[parent].right = box
            self.boxes[box].low[dim] = value
            if right-(mid+1) > 1:
                # At least 3 points remaining in right box?
                # ...request a right-split
                task_id += 1
                task_nodes[task_id] = box
                task_dims[task_id] = next_dim
        free(task_nodes)
        free(task_dims)

    cdef _init_root_box(self):
        cdef:
            unsigned int n
        for n in range(self.ndim):
            self.boxes[0].low[n] = -HUGE_VAL
            self.boxes[0].high[n] = HUGE_VAL
        self.boxes[0].low_idx = 0
        self.boxes[0].high_idx = self.size-1
        self.boxes[0].parent = 0
        self.boxes[0].left = 0
        self.boxes[0].right = 0

    cdef _fork_at(self, unsigned int box, unsigned int parent,
            unsigned int low_idx, unsigned int high_idx):
        cdef:
           unsigned int n
        self.boxes[box].parent = parent
        self.boxes[box].left = 0
        self.boxes[box].right= 0
        self.boxes[box].low_idx = low_idx
        self.boxes[box].high_idx = high_idx
        for n in range(self.ndim):
            self.boxes[box].low[n] = self.boxes[parent].low[n]
            self.boxes[box].high[n] = self.boxes[parent].high[n]

    cdef _allocate_memory(self):
        cdef:
            int m=1
            int nboxes
        # I. Boxes
        # Find first power of 2 equal to/greater than size...
        while m < self.size:
            m *= 2
        self.nboxes = 2*self.size - (m>>1)
        if self.nboxes > m:
            self.nboxes = m
        self.nboxes -= 1
        # Make sure we have at least enough room for root box
        if self.nboxes == 0:
            self.nboxes = 1
        self.boxes = <KDTreeNode*> PyMem_Malloc(
                self.nboxes * sizeof(KDTreeNode))
        for m in range(self.nboxes):
            self.boxes[m].low = <np.float_t*> PyMem_Malloc(
                    self.ndim * sizeof(np.float_t))
            self.boxes[m].high = <np.float_t*> PyMem_Malloc(
                    self.ndim * sizeof(np.float_t))
        # II. Indices
        self._idx = <unsigned int*> PyMem_Malloc(
                self.size * sizeof(unsigned int))

    def __dealloc__(self):
        cdef:
            int m
        for m in range(self.nboxes):
            PyMem_Free(self.boxes[m].low)
            PyMem_Free(self.boxes[m].high)
        PyMem_Free(self.boxes)
        PyMem_Free(self._idx)


@cython.boundscheck(False)
@cython.initializedcheck(False)
cdef unsigned int _partition_idx(unsigned int at, unsigned int *idx,
        double* values, unsigned int size):
    """
    Quicksort(in-place) permutation of idx such that
        values[idx[0..at-1]] <= values[idx[at]] <= values[idx[at+1..size-1]]
    * function calls leave `values` untouched
    * uses middle element as pivot
    """

    cdef:
        # Pivot
        unsigned int pivot # Current pivot
        double pivot_value  # To be compared
        # Indices of indices...
        unsigned int left, mid, right
        unsigned int left_gtr  # First-greater-from-left
        unsigned int right_lwr  # First-lower-from-right

    left = 0
    right = size-1

    while True:
        pivot = left+1
        if (right <= pivot):
            if (right == pivot and values[idx[right]] < values[idx[left]]):
                idx[left], idx[right] = idx[right], idx[left]
            return idx[at]
        else:
            # I. Warm-up
            # Send middle value to pivot
            mid = (left+right) >> 1  # same as floor((left+right)*0.5)
            idx[mid], idx[pivot] = idx[pivot], idx[mid]
            # Make sure that...
            #   values[idx[left]]<values[idx[pivot]]<values[idx[right]]
            if values[idx[left]] > values[idx[right]]:
                idx[left], idx[right] = idx[right], idx[left]
            if values[idx[pivot]] > values[idx[right]]:
                idx[pivot], idx[right] = idx[right], idx[pivot]
            if values[idx[left]] > values[idx[pivot]]:
                idx[left], idx[pivot] = idx[pivot], idx[left]
            # II. Partitioning
            pivot_value = values[idx[pivot]]
            left_gtr = pivot  # From left
            right_lwr = right  # From right
            while True:
                for left_gtr in range(left_gtr+1, size):
                    if values[idx[left_gtr]] >= pivot_value: break
                for right_lwr in range(right_lwr-1, -1, -1):
                    if values[idx[right_lwr]] <= pivot_value: break
                if left_gtr > right_lwr:
                    # Cursors intersecting...
                    break  # Done partitioning
                else:
                    # Push lower to the left and greater to the right
                    idx[left_gtr], idx[right_lwr] = \
                            idx[right_lwr], idx[left_gtr]
            # Now put back the pivot to the right of right_lwr
            # since value[idx[right_lwr]] is lower than or equal to
            #   pivot_value
            idx[pivot], idx[right_lwr] = idx[right_lwr], idx[pivot]
            # III. Tighten cursors
            # At this point,
            #   (1) value[idx[right_lwr...size-1]] >= pivot_value
            #   (2) value[idx[0..left_gtr-1]] <= pivot_value
            if at <= right_lwr:
                # Left partition needs more sorting...
                #   .. tighten right bound
                right = right_lwr-1
            # Left:
            if at >= right_lwr:
                # Right partition needs more sorting...
                #   .. tighten left bound
                left = left_gtr
