%Solves for the one positive root of the Rrs IOP relationship
%see Wang, Boss, and Roesler, 2005, Applied optics.
function [V] = v(L3, L4, Rrs);
x = [-L3+sqrt(L3.^2+4*L4*Rrs)]./(2*L4);
V = 1-1./x;