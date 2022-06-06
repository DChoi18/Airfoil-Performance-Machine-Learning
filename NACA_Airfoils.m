function [x,y] = NACA_Airfoils(m,p,t,c,N)
% function to compute x,y coordinates of a NACA 4-digit series airfoil
% Inputs:
%         m = max camber 
%         p = location of max camber
%         t = thickness chord
%         c = chord length
%         N = number of panels
% Output:
%         x = x-location of boundary points of airfoil
%         y = y-location of boundary points of airfoil
%
% Author: Derrick Choi
% Collaborators: Aneesh Balla

%% Discretize the chord
%use a circle centered at midchord 
radius = c/2;
%divide circle into N arcs of equal length
theta = 0:2*pi/N:2*pi; %angles between arcs for half of the circle
%project on to chord line
x_disc = radius*cos(theta);
%shift xpoints so that the leading edge is at x = 0;
x_disc = c/2+x_disc;
%% Find corresponding x and y-boundary points
x_c = x_disc/c; %normalize x points by chord
yt = t/0.2*c*(0.2969*sqrt(x_c)-0.1260*x_c-0.3516*x_c.^2+0.2843*x_c.^3-0.1036*x_c.^4);
%preallocate memory for camber line and its slope
yc = zeros(1,length(x_disc));
dyc_dx = zeros(1,length(x_disc));
%Define iterator variable
i = 1;
%Generate camber line and the camber line derivative at discrete points
while i <= N
    %for front of airfoil less than position of max camber
    if x_disc(i) >= 0 && x_disc(i) <= p*c
        yc(i) = m*x_disc(i)/p^2*(2*p-x_c(i));
        dyc_dx(i) = 2*m/(p^2)*(p-x_c(i));
    else
    %for back of airfoil greater than position of max camber
        yc(i) = m*(c-x_disc(i))/(1-p)^2*(1+x_c(i)-2*p);
        dyc_dx(i) = 2*m/(1-p)^2*(p-x_c(i));
    end
    i = i+1; %update iterator
end
%resolve singularities in derivative (occurs when m = 0 and p = 0)
dyc_dx(isnan(dyc_dx)) = 0;
yc(isnan(yc)) = 0;

%Define angle of camber line relative to horizontal at given x point
zeta = atan2(dyc_dx,1);

%coordinates of upper and lower surfaces of the airfoil (the indexing used
%is so that coordinates are not repeated) 
xu = x_disc(ceil(N/2)+1:end)-yt(ceil(N/2)+1:end).*sin(zeta(ceil(N/2)+1:end));
xl = x_disc(1:ceil(N/2))+yt(1:ceil(N/2)).*sin(zeta(1:ceil(N/2)));
yu = yc(ceil(N/2)+1:end)+yt(ceil(N/2)+1:end).*cos(zeta(ceil(N/2)+1:end));
yl = yc(1:ceil(N/2))-yt(1:ceil(N/2)).*cos(zeta(1:ceil(N/2)));

%Output x-coordinates and y-coordinates from trailing edge to leading edge
%in clockwise manner around airfoil surface
x = [xl,xu];
y = [yl,yu];

end