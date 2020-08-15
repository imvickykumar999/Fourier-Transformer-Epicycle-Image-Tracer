import numpy as np
from itertools import chain

class Fourier(object):
    def __init__(self, n_approx = 1000, coord_1 = None, coord_2 = None):
        if coord_1 is not None:
            temp = coord_1[:,:,0] + 1j * coord_1[:,:,1]
            self.complex_coord_1 = temp.reshape(temp.shape[0])
        if coord_2 is not None:
            temp = coord_2[:,:,0] + 1j * coord_2[:,:,1]
            self.complex_coord_2 = temp.reshape(temp.shape[0])
            if self.complex_coord_2.size > self.complex_coord_1.size:
                self.complex_coord_1 = np.hstack((self.complex_coord_1, np.full((self.complex_coord_2.size - self.complex_coord_1.size), self.complex_coord_1[-1], dtype = np.complex_)))
            elif self.complex_coord_1.size > self.complex_coord_2.size:
                self.complex_coord_2 = np.hstack((self.complex_coord_2, np.full((self.complex_coord_1.size - self.complex_coord_2.size), self.complex_coord_2[-1], dtype = np.complex_)))
        else:
            self.complex_coord_2 = None

        self.n_approx = self.complex_coord_1.size//2 if n_approx > self.complex_coord_1.size//2  else n_approx

    def get_circles(self, mode=1):
        if self.complex_coord_1 is not None and self.complex_coord_2 is not None:
            return self.get_two_circles_two_images()
        elif mode == 2:
            return self.get_two_circles_one_image()
        return self.get_one_circle_one_image()
        
    def get_one_circle_one_image(self):
        period = self.complex_coord_1.size
        time   = np.arange(period)
        circles_loc = np.zeros((2*(self.n_approx-1), period), dtype = np.complex_)
        circles_rad = np.zeros((2*(self.n_approx-1)), dtype = np.float_)

        for idx, multiple in enumerate(chain(range(-self.n_approx+1, 0), range(1, self.n_approx))):
            cn = self.cn(time, period, multiple, self.complex_coord_1)
            circles_rad[idx] = np.absolute(cn)
            circles_loc[idx, :] = self.polar_locations(time, period, multiple, cn)

        order = np.argsort(circles_rad)[::-1]
        circles_loc = circles_loc[order]
        circles_rad = circles_rad[order]
        circles_loc = np.add.accumulate(circles_loc, 0)
                        
        return period, (circles_rad,), (circles_loc,)

    def get_two_circles_one_image(self):
        period = self.complex_coord_1.size
        time   = np.arange(period)
        circles_loc_1 = np.zeros((2*(self.n_approx - 1), period), dtype = np.complex_)
        circles_rad_1 = np.zeros((2*(self.n_approx - 1)), dtype = np.float_)
        circles_loc_2 = np.zeros((2*(self.n_approx - 1), period), dtype = np.complex_)
        circles_rad_2 = np.zeros((2*(self.n_approx - 1)), dtype = np.float_)

        for idx, multiple in enumerate(range(1, self.n_approx)):

            cn_1 = self.cn(time, period, multiple, self.complex_coord_1.real)
            an_1, bn_1 = cn_1.real, cn_1.imag

            circles_rad_1[idx] = np.absolute(an_1)
            circles_rad_1[idx+self.n_approx-1] = np.absolute(bn_1)

            circles_loc_1[idx, :] = self.cartesian_locations(time, period, multiple, an_1)
            circles_loc_1[idx+self.n_approx-1, :] = self.cartesian_locations(time, period, multiple, bn_1)
            circles_loc_1[idx+self.n_approx-1, :] = circles_loc_1[idx + self.n_approx - 1, :].imag + 1j * circles_loc_1[idx + self.n_approx - 1, :].real
            
            cn_2 = self.cn(time, period, multiple, self.complex_coord_1.imag)
            an_2, bn_2 = cn_2.real, cn_2.imag

            circles_rad_2[idx] = np.absolute(bn_2)
            circles_rad_2[idx+self.n_approx-1] = np.absolute(an_2)
            
            circles_loc_2[idx, :] = self.cartesian_locations(time, period, multiple, bn_2)
            circles_loc_2[idx+self.n_approx-1, :] = self.cartesian_locations(time, period, multiple, an_2)
            circles_loc_2[idx+self.n_approx-1, :] = circles_loc_2[idx + self.n_approx - 1, :].imag + 1j * circles_loc_2[idx + self.n_approx - 1, :].real

        order_1 = np.argsort(circles_rad_1)[::-1]
        circles_loc_1 = circles_loc_1[order_1]
        circles_rad_1 = circles_rad_1[order_1]
        order_2 = np.argsort(circles_rad_2)[::-1]
        circles_loc_2 = circles_loc_2[order_2]
        circles_rad_2 = circles_rad_2[order_2]
        
        circles_loc_1 = np.add.accumulate(circles_loc_1, 0)
        circles_loc_2 = np.add.accumulate(circles_loc_2, 0)

        return period, (circles_rad_1, circles_rad_2), (circles_loc_1, circles_loc_2)

    def get_two_circles_two_images(self):
        period = self.complex_coord_1.size
        time   = np.arange(period)
        circles_loc_1 = np.zeros((2*(self.n_approx-1), period), dtype = np.complex_)
        circles_rad_1 = np.zeros((2*(self.n_approx-1)), dtype = np.float_)
        circles_loc_2 = np.zeros((2*(self.n_approx-1), period), dtype = np.complex_)
        circles_rad_2 = np.zeros((2*(self.n_approx-1)), dtype = np.float_)
            
        for idx, multiple in enumerate(chain(range(-self.n_approx+1, 0), range(1, self.n_approx))):
            cn_1 = self.cn(time, period, multiple, self.complex_coord_1.real + 1j * self.complex_coord_2.imag)
            cn_2 = self.cn(time, period, multiple, self.complex_coord_2.real + 1j * self.complex_coord_1.imag)

            circles_rad_1[idx] = np.absolute(cn_1)
            circles_rad_2[idx] = np.absolute(cn_2)
                  
            circles_loc_1[idx, :] = self.polar_locations(time, period, multiple, cn_1)              
            circles_loc_2[idx, :] = self.polar_locations(time, period, multiple, cn_2)
                                 
        order_1 = np.argsort(circles_rad_1)[::-1]
        circles_loc_1 = circles_loc_1[order_1]
        circles_rad_1 = circles_rad_1[order_1]
        order_2 = np.argsort(circles_rad_2)[::-1]
        circles_loc_2 = circles_loc_2[order_2]
        circles_rad_2 = circles_rad_2[order_2]
                                 
        circles_loc_1 = np.add.accumulate(circles_loc_1, 0)
        circles_loc_2 = np.add.accumulate(circles_loc_2, 0)

        return period, (circles_rad_1, circles_rad_2), (circles_loc_1, circles_loc_2)

    def cn(self, time, period, multiple, coordinates):
        c = coordinates * np.exp(-1j * (2*multiple*np.pi/period) * time)
        return c.sum() / period

    def polar_locations(self, time, period, multiple, fourier_coeff):
        return np.absolute(fourier_coeff) * np.exp(1j * ((2*multiple*np.pi/period) * time + np.angle(fourier_coeff)))

    def cartesian_locations(self, time, period, multiple, fourier_coeff):
        return fourier_coeff * np.exp(1j * ((2*multiple*np.pi/period) * time))
