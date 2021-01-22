import numpy as np
import scipy.signal as sp
import scipy.linalg as la
import matplotlib.pyplot as plt
from math import pi
import copy
import sys


class Box:
    '''
    This class is a helper class for running a full Bayesian
    filter in 2d.  When you have a PDF, you can store the
    outside limits of the box in this class.  It also has 
    two helper functions that generate a new box from the
    convolution of two boxes (propagate step), and generates a "union" box
    for when you do a multiplication of two boxes (update step)
    '''

    def __init__(self, x_low=0, x_high=0, y_low=0, y_high=0, dx=None):
        '''
        If dx is None, then x_low... is simply stored in limits
        If not, then x_low ... are "rounded" to the closest 
        dx divisible value.  This is needed if conv and mult
        are going to be used a bit later on
        '''
        assert x_high >= x_low, "Invalid limits for Box"
        assert y_high >= y_low, "Invalid limits for Box"
        self.limits = np.zeros((2, 2))
        if dx is not None:
            x_low = np.rint(x_low / dx) * dx
            x_high = np.rint(x_high / dx) * dx
            y_low = np.rint(y_low / dx) * dx
            y_high = np.rint(y_high / dx) * dx
        self.limits[0, 0] = x_low
        self.limits[0, 1] = x_high
        self.limits[1, 0] = y_low
        self.limits[1, 1] = y_high

    def gen_xs(self, dx):
        '''
        Takes the limits stored in this class and generates an 
        m X n X 2 array with all the points in the box.  
        m will be determined by dx and the size of the y limits
        n will be determined by dx and the size of the x limits

        Args:
            dx: A single float denoting the size of each "box"
                to be generated
        
        Returns:
            A 3d numpy array, where 2 is the last dimension
        '''
        x, y = self.gen_axes(dx)
        return np.array(np.meshgrid(x, y, indexing='ij')).T

    def gen_axes(self, dx):
        '''
        Takes the limits stored in this class and generates 2
        arrays corresponding to the x and y values for this Box.

        Args:
            dx: A single float denoting the size of each "box"
                to be generated
        
        Returns:
            A tuple with of two 1-D numpy arrays, (x,y)
        '''

        x = np.arange(self.limits[0, 0], self.limits[0, 1] + dx / 10., dx)
        y = np.arange(self.limits[1, 0], self.limits[1, 1] + dx / 10., dx)
        return (x, y)

    def conv(self, box2):
        '''
        This function takes in two boxes and returns a box
        with the correct limits if it is run through a convolution
        with the 'full' option.  This means the output will
        be bigger than either of the inputs

        Args:
            box2: input of type "Box"

        Returns:
            A Box class
        '''
        # Find the middle of the new box
        mid1 = np.sum(self.limits, axis=1) / 2.
        mid2 = np.sum(box2.limits, axis=1) / 2.
        new_mid = mid1 + mid2

        # And how big the box will be
        delta1 = self.limits[:, 1] - mid1
        delta2 = box2.limits[:, 1] - mid2
        new_delta = delta1 + delta2

        return Box(new_mid[0] - new_delta[0], new_mid[0] + new_delta[0],
                   new_mid[1] - new_delta[1], new_mid[1] + new_delta[1])

    def mult(self, box2, dx=None):
        '''
        This function takes in two boxes and returns the
        "union" between them, i.e. only the places where there 
        is data from both boxes

        Args:
            self, box2:  Two objects of type "Box"

        Returns:
            a Box class 
        '''
        min_x = self.limits[0, 0]
        if box2.limits[0, 0] > min_x:
            min_x = box2.limits[0, 0]

        max_x = self.limits[0, 1]
        if box2.limits[0, 1] < max_x:
            max_x = box2.limits[0, 1]

        min_y = self.limits[1, 0]
        if box2.limits[1, 0] > min_y:
            min_y = box2.limits[1, 0]

        max_y = self.limits[1, 1]
        if box2.limits[1, 1] < max_y:
            max_y = box2.limits[1, 1]

        return Box(min_x, max_x, min_y, max_y)

    def get_sub_array(self, array, dx, small_box):
        '''
        This function can be used in conjunction with mult.  When
        you have a small box within the current class, and an array
        (with dx) that corresponds with this box, you can pass in
        a smaller box and get back an array for that smaller area.

        Args:
            array:  the bigger array size to get data from.  Should correspond
                in size to the Box class and dx
            dx: the spacing of the array
            small_box:  the smaller box to fit inside this box
        
        Returns:
            an numpy array that is <= the array

        WARNING:  I don't check that array and dx correspond with 
        box.  They should.  Also remember y is the first entry,
        x is the second index.  (So it makes pretty pictures.)
        '''
        assert np.all(small_box.limits[:, 0] >= self.limits[:, 0]), "small box must fit within this box"
        assert np.all(small_box.limits[:, 1] <= self.limits[:, 1]), "small box must fit within this box"

        # I do this in two steps as that is the only way I found that I could properly
        # grab sub-blocks out of arrays using numpy.  Maybe there is a better way, but...

        # First, grab the right rows... (y values)
        y_low = int(np.rint((small_box.limits[1, 0] - self.limits[1, 0]) / dx))
        y_high = int(np.rint((small_box.limits[1, 1] - self.limits[1, 0]) / dx) + 1)
        tmp = array[y_low:y_high]

        # Now grab the cols
        x_low = int(np.rint((small_box.limits[0, 0] - self.limits[0, 0]) / dx))
        x_high = int(np.rint((small_box.limits[0, 1] - self.limits[0, 0]) / dx) + 1)
        return tmp[:, x_low:x_high]


def p_normal_2d(xs, mean, cov):
    '''
    This function takes in a bunch of 2d locations (xs) and a mean and
    covariance and puts in the correct probability values.

    Args:
        xs:  A []x2 numpy array with locations at which to compute the 
            probability values
        mean: A (2,) numpy array with the mean of the Gaussian distribution
        cov: A 2x2 numpy array that has the covariance of the normal

    Returns:
        A []-sized numpy array with a bunch of probability values, where [] is
            the shape of xs, less the last part of shape

    WARNINGS:
        I do no error checking on cov.  If it is not positive definite, or symmetric, or....
        Nor do I do any error checking on the size of xs, mean, etc.
    '''

    scale_const = 1 / np.sqrt(la.det(2 * pi * cov))
    inv_cov = la.inv(cov)

    num_points = 1
    for dim in xs.shape[:-1]:
        num_points *= dim
    internal_xs = np.reshape(xs, (num_points, 2)) - mean
    internal_res = np.zeros(num_points)
    for i, x in enumerate(internal_xs):
        internal_res[i] = -0.5 * x.dot(inv_cov.dot(x))
    internal_res = np.exp(internal_res)
    internal_res *= scale_const

    return np.reshape(internal_res, xs.shape[:-1])


class ForwardProb:
    # Create the "dynamics" array.
    # This will have three Gaussians.  Left, Straight, and right
    # The probability of each Gaussian occurring and the mean of that Gaussian is
    # setup in the parameters below. Also, the covariance of each of the Gaussians
    # is defined by cov
    # We define x as right(positive)-left and y as forward(positive)-backwards

    p_left = .3
    left_loc = np.array([-3, 4])

    p_right = .3
    right_loc = np.array([3, 4])

    p_center = .4
    center_loc = np.array([0, 5])

    cov = np.eye(2) * 0.1

    # When creating a big array to hold the probability values, need to know
    # where the corners are
    box = Box()  # This will be the lower left [0] and upper right[1] values

    # These next two values essentially correspond with box, but are the range of them
    # so that they can be used for plotting, generating data, etc.

    def __init__(self, dx):
        # Make the probability values used to move the real location (randomly)
        self.left_test = self.p_left
        self.right_test = self.p_left + self.p_right
        self.apply_noise = la.cholesky(self.cov)

        # Make the probability mask to be convolved... (p_process)
        # First, let's figure out how big of a box I need
        # Around each Gaussian need to have at least a +/- box of size...
        num_sd = 5
        x_range = num_sd * np.sqrt(self.cov[0, 0])
        y_range = num_sd * np.sqrt(self.cov[1, 1])

        # Search for min and max for big box
        x_min = self.left_loc[0] - x_range
        x_max = self.left_loc[0] + x_range
        y_min = self.left_loc[1] - y_range
        y_max = self.left_loc[1] + y_range

        # Helper functions to create boxes
        new_min = lambda prior_min, center, delta: prior_min if prior_min < (center - delta) else center - delta
        new_max = lambda prior_max, center, delta: prior_max if prior_max > (center + delta) else center + delta

        x_min = new_min(x_min, self.center_loc[0], x_range)
        x_min = new_min(x_min, self.right_loc[0], x_range)
        x_max = new_max(x_max, self.center_loc[0], x_range)
        x_max = new_max(x_max, self.right_loc[0], x_range)

        y_min = new_min(y_min, self.center_loc[1], y_range)
        y_min = new_min(y_min, self.right_loc[1], y_range)
        y_max = new_max(y_max, self.center_loc[1], y_range)
        y_max = new_max(y_max, self.right_loc[1], y_range)

        self.box = Box(x_min, x_max, y_min, y_max, dx)

        # Generate the probabilities within that big box
        xs = self.box.gen_xs(dx)
        left_p = p_normal_2d(xs, self.left_loc, self.cov) * self.p_left
        right_p = p_normal_2d(xs, self.right_loc, self.cov) * self.p_right
        straight_p = p_normal_2d(xs, self.center_loc, self.cov) * self.p_center
        self.p_process = left_p + right_p + straight_p

    def time_prop(self, curr_x):
        '''
        Take in curr_x and, using the probabilities stored in this
        class, propagate the robot forward in time.  This consists of 
        deciding which Gaussian it will go to (left, center, right) and
        also applying noise from that Gaussian

        Args:
            curr_x: a (2,) sized np.array that holds the current robot location

        Returns: the next location of the robot (another (2,) array)
        '''
        sample = np.random.rand()
        next_x = curr_x
        if sample < self.left_test:
            next_x += self.left_loc
        elif sample < self.right_test:
            # going right
            next_x += self.right_loc
        else:
            next_x += self.center_loc
        return next_x + self.apply_noise.dot(np.random.randn(2))


class MeasureProb:
    # Measurement noise is going to be zero-mean Gaussian with the following covariance

    def __init__(self, dx, cov=np.eye(2) * 2, mean=np.zeros(2)):
        self.cov = cov
        # First, turn cov into something that can be sampled
        self.apply_noise = la.cholesky(self.cov)

        # Second, generate a mask for probability
        # Your assignment: generate a box that goes out 5 standard deviations
        num_sd = 5

        x_std = np.sqrt(self.cov[0, 0]) * num_sd
        y_std = np.sqrt(self.cov[1, 1]) * num_sd

        # replace this line for sure!
        self.box = Box(mean[0] - x_std, mean[0] + x_std, mean[1] - y_std, mean[1] + y_std, dx=dx)

        xs = self.box.gen_xs(dx)
        # This generates the PDF
        self.p_array = p_normal_2d(xs, mean, self.cov)

    def gen_meas(self, true_loc):
        # Your assignment: Sample from the PDF.  Return a (2,) sized np.array
        sample = np.dot(self.apply_noise, np.random.rand(2, )) + true_loc
        return sample


if __name__ == "__main__":
    # Rather than commenting and uncommenting code, this denotes different
    # parts of the code that can run.  These correspond with
    # Propagate only, 1 step (p1)
    # Propagate only, multiple steps (pm)
    # Full Bayes filter, (bf)

    full_to_run = ['ut1', 'p1', 'pm', 'bf']
    print("Usage:  can pass in any of these values ", full_to_run)
    print("        Or type use _full_ to run everything")

    to_run = ['bf']
    if len(sys.argv) > 1:
        if 'full' in sys.argv[1:]:
            to_run = full_to_run
        else:
            to_run = sys.argv[1:]
    print("Going to run", to_run)

    dx = 1 / 20.
    fp = ForwardProb(dx)

    if 'ut1' in to_run:
        # Test a few things out with Box
        tst_box = Box(0, 1, 0, 2)
        x, y = tst_box.gen_axes(dx)
        assert np.allclose(x, np.arange(0, 1 + dx / 10, dx)), "Box.gen_axes failed"
        assert np.allclose(y, np.arange(0, 2 + dx / 10, dx)), "Box.gen_axes failed"
        # Draw a simple Gaussian to test gen_axes
        tst_box = Box(-3, 3, -3, 3)
        xs = tst_box.gen_xs(dx)
        gauss_prob = p_normal_2d(xs, np.array([0, 0]), np.eye(2) * .25)
        curr_x, curr_y = tst_box.gen_axes(dx)
        plt.contourf(curr_x, curr_y, gauss_prob)
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.text(0, 0, "Gaussian distribution!")
        plt.show()
        # Now test conv
        tst_box1 = Box(0, 1, 2, 3)
        tst_box2 = Box(-.5, .5, -.5, .5)  # Should just add .5 to everything
        new_box = tst_box1.conv(tst_box2)
        should_be = np.array([[-.5, 1.5], [1.5, 3.5]])
        assert np.allclose(new_box.limits, should_be), 'box.conv() failed'

        # Now test mult
        tst_box1 = Box(0, 3, 1, 2)
        tst_box2 = Box(1, 2, 0, 3)
        new_box = tst_box1.mult(tst_box2)
        should_be = np.array([[1, 2], [1, 2]])
        assert np.allclose(new_box.limits, should_be), 'box.mult() failed'

        # Need to test get_sub_array
        xs = tst_box1.gen_xs(dx)
        sub_box = Box(1, 2, 1, 1.5)
        tst_array = tst_box1.get_sub_array(xs, dx, sub_box)
        assert np.allclose(tst_array, sub_box.gen_xs(dx)), 'box.get_sub_array failed'

        print('Got this far. Box must be (at least sort-of) working')

    if 'p1' in to_run:
        np.random.seed(55)
        # generate a starting location
        p0 = MeasureProb(dx, np.eye(2) * .1)
        start_p = p0.p_array
        true_start = p0.gen_meas(np.zeros(2))
        # Propagate the true location forward in time
        true_loc = fp.time_prop(true_start)
        # Now propagate the PDF
        out_p = sp.convolve2d(start_p, fp.p_process) * dx * dx
        curr_box = p0.box.conv(fp.box)
        # Plot the result
        curr_x, curr_y = curr_box.gen_axes(dx)
        plt.contourf(curr_x, curr_y, out_p)
        plt.colorbar()
        plt.scatter(true_loc[0], true_loc[1], color='r')
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.show()

    if 'pm' in to_run:
        n_steps = 4
        p0 = MeasureProb(dx, np.eye(2) * .1)
        curr_p = p_normal_2d(p0.box.gen_xs(dx), np.array([0, 0]), np.eye(2) * .1)
        curr_box = copy.copy(p0.box)
        true_start = p0.gen_meas(np.zeros(2))
        # Your assignment:  Propagate forward in time n_steps
        # result should be in curr_p and curr_box

        for i in range(n_steps):
            true_loc = fp.time_prop(true_start)
            curr_p = sp.convolve2d(curr_p, fp.p_process) * dx * dx
            curr_box = curr_box.conv(fp.box)
            true_start = curr_box.gen_meas(true_loc)

        # Plot it.
        curr_x, curr_y = curr_box.gen_axes(dx)
        plt.contourf(curr_x, curr_y, curr_p)
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.show()

    if 'bf' in to_run:
        np.random.seed(828)
        n_steps = 5
        # initialize truth and the curr_PDF
        p0 = MeasureProb(dx, np.eye(2) * .1)
        mp = MeasureProb(dx)
        curr_p = p_normal_2d(p0.box.gen_xs(dx), np.array([0, 0]), np.eye(2) * .1)
        true_loc = p0.gen_meas(np.array([0, 0]))
        curr_box = copy.copy(p0.box)
        print(f'true location at time 0 is {true_loc}')

        for i in range(n_steps):
            # Your Assignment:  Implement a Bayesian filter & simulation

            # YA: Time propagate reality, put location in true_loc
            true_loc = fp.time_prop(true_loc)
            print(f'true location at time {i + 1} is {true_loc}')

            # YA: Time propagate PDF, put it in curr_p and curr_box
            curr_p = sp.convolve2d(curr_p, fp.p_process)
            curr_box = curr_box.conv(fp.box)
            # YA: Get a measurement and put it in curr_meas
            curr_meas = mp.gen_meas(true_loc)
            curr_meas_p = MeasureProb(dx, mean=curr_meas)
            mult_box = curr_box.mult(curr_meas_p.box)
            prior_p = curr_box.get_sub_array(curr_p, dx, mult_box)
            meas_p = curr_meas_p.box.get_sub_array(curr_meas_p.p_array, dx, mult_box)
            curr_p = prior_p * meas_p
            curr_box = mult_box
            curr_p /= np.sum(curr_p) * dx * dx
            # YA: Apply measurement to current_array.  Put result in curr_p and curr_box

            ##Create measurement PDF
            ##Create smaller box that is union of prior and measurement
            ##Grab respective boxes and multiply them
            ## Normalize multiplied output

            # Create new figure and draw
            plt.figure()
            curr_x, curr_y = curr_box.gen_axes(dx)
            plt.contourf(curr_x, curr_y, curr_p)
            plt.colorbar()
            plt.scatter(true_loc[0], true_loc[1], color='r', s=3)
            plt.title(f'Timestep {i + 1}')
            plt.savefig(f'{i + 1}.png')
            ax = plt.gca()
            ax.set_aspect('equal')
            plt.show(block=False)
            plt.pause(.01)
        print(f"Bayes filtering done for {n_steps} timesteps.  Hit return to finish...")
        input()
        plt.close('all')
