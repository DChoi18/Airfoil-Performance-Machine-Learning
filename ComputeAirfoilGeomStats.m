function geom = ComputeAirfoilGeomStats(x,yu,yl)
% function to compute parameters that characterize geometry of airfoil
%
% Inputs:
%        x = x-coordinates of airfoil
%        yu = upper surface airfoil coordinates
%        yl = lower surface airfoil coordinates
% Outputs:
%        geom = struct variable with following fields
%            mean_camber = camber at each x location
%            t = thickness at each x location
%            pos_c = position of max camber
%            pos_t = position of max thickness
%            t_max = max thickness normalized by chord
%            max_camb = max camber normalized by chord
%
% Author: Derrick Choi

% chord
c = max(x)-min(x);

% camber and thickness distributions
mean_camber = mean([yu yl],2);
t = (abs(yu-yl));

% Do a natural spline fit to get more data points on the airfoil
[xspline_u,yspline_u] = SplineInterp(x,yu,0,'natural');
[~,yspline_l] = SplineInterp(x,yl,0,'natural');

% max camber and thickness and their locations
[t_max,tmax_id] = max(abs(yspline_u-yspline_l));
[max_camb,camb_max_id] = max(mean([yspline_u;yspline_l],1));

pos_c = xspline_u(camb_max_id);
pos_t = xspline_u(tmax_id);

% non-dimensionalize by chord
pos_t = pos_t/c;
pos_c = pos_c/c;
mean_camber = mean_camber/c;
t = t/c;

geom = struct('mean_camber',mean_camber,'t',t,'t_max',t_max,'max_camb',max_camb,'pos_t',pos_t,'pos_c',pos_c);

end