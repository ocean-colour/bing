%setup the array of known parameter for the inversion
%see Wang, Boss, and Roesler, 2005, Applied optics.

function [h] = Array_h(a_sea_water, bb_sea_water, V3);
h = - ([a_sea_water'] + [bb_sea_water'].*[V3]);