function [x_interp,y_interp] = SplineInterp(xi,yi,yi_prime,mode)
% function to interpolate a function with cubic spline interpolation using
% natural boundaries or clamped boundaries. (code follows pseudocode from a
% numerical analysis textbook)
%
% Inputs: 
%         xi = nodes
%         yi = data at nodes
%         yi_prime = derivative data at boundaries (not used if natural
%                    boundaries are enforced)
%         mode = 'natural' or 'clamped' boundaries
% Outputs:
%         x_interp = x-values for evaluation of cubic spline
%         y_interp = cubic spline interpolated values at x_interp
% Author: Derrick Choi

% preallocate
x_interp = zeros(1,100*length(diff(xi)));
y_interp = zeros(1,100*length(diff(xi)));
% Determine type of spline interpolation
switch mode
    case 'natural'
        %form linear system
        h = diff(xi);
        npts = length(xi);
        A = zeros(npts,npts);
        A(1,1) = 1;
        A(npts,npts) = 1;
        f = zeros(npts,1);
        for i = 2:npts-1
            for j = 1:npts-1
                if i == j
                   A(i,j) = (h(j)+h(j-1))/3;
                   A(i,j-1) = h(j-1)/6;
                   A(i,j+1) = h(j)/6;
                end
            end
            f(i) = (yi(i+1)-yi(i))/h(i) - (yi(i)-yi(i-1))/h(i-1);
        end
        %solve linear system
        M = A\f;
        %form the cubic spline over an interval and evaluate for 100 points
        %in the interval
        for i = 1:length(M)-1
           C = yi(i)/h(i)-h(i)/6*M(i);
           D = yi(i+1)/h(i)-h(i)/6*M(i+1);
           S = @(x) M(i)*(xi(i+1)-x).^3/(6*h(i))+M(i+1)*(x-xi(i)).^3/(6*h(i)) +C*(xi(i+1)-x)+D*(x-xi(i));
           x_interp(1+(i-1)*200:i*200) = linspace(xi(i),xi(i+1),200);
           y_interp(1+(i-1)*200:i*200) = S(x_interp(1+(i-1)*200:i*200));
        end
    case 'clamped'
        %form linear system
        h = diff(xi);
        npts = length(xi);
        
        A = zeros(npts,npts);
        A(1,1) = h(1)/3;
        A(1,2) = h(1)/6;
        A(npts,npts) = h(end)/3;
        A(npts,npts-1) = h(end)/6;
   
        f = zeros(npts,1);
        f(1) = -yi_prime(1)+(yi(2)-yi(1))/h(1);
        f(npts) = yi_prime(2)-(yi(npts)-yi(npts-1))/h(end);
        for i = 2:npts-1
            for j = 1:npts-1
                if i == j
                   A(i,j) = (h(j)+h(j-1))/3;
                   A(i,j-1) = h(j-1)/6;
                   A(i,j+1) = h(j)/6;
                end
            end
            f(i) = (yi(i+1)-yi(i))/h(i) - (yi(i)-yi(i-1))/h(i-1);
        end
        %solve linear system
        M = A\f;
        %form the cubic spline over an interval and evaluate for 100 points
        %in the interval
        for i = 1:length(M)-1
           C = yi(i)/h(i)-h(i)/6*M(i);
           D = yi(i+1)/h(i)-h(i)/6*M(i+1);
           S = @(x) M(i)*(xi(i+1)-x).^3/(6*h(i))+M(i+1)*(x-xi(i)).^3/(6*h(i)) +C*(xi(i+1)-x)+D*(x-xi(i));
           x_interp(1+(i-1)*100:i*100) = linspace(xi(i),xi(i+1),100);
           y_interp(1+(i-1)*100:i*100) = S(x_interp(1+(i-1)*100:i*100));
        end
end

end