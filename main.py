from Image import Image
from Fourier import Fourier
from Plot import Plot

im_1 = Image("images/flag.png", (200, 200))
im_2 = Image("images/text.png", (200, 200))

path_1 = im_1.sort()
path_2 = im_2.sort()

period_3, tup_circle_rads_3, tup_circle_locs_3 = Fourier(n_approx = 1000, coord_1 = path_1, coord_2 = path_2).get_circles()
Plot(period_3, tup_circle_rads_3, tup_circle_locs_3, speed = 10).plot(close_after_animation = False)
