run /home/mathieu/Documents/MATLAB/SDPNAL+v1.0/startup.m

A1 = cell2mat([D1{:}]);
A2 = cell2mat([D2{:}]);
sz1 = size(A1);
sz2 = size(A2);
s1 = sqrt(sz1(2));
s2 = sqrt(sz2(2));

[X,Gamma,b,y,info,runhist] = gh_sdpnal_D(double(reshape(A1,s1,s1)), double(reshape(A2,s2,s2)));
N=size(X,1);
x=X(N, 1:N-1);
n=floor(sqrt(N-1));
maps=birkhoff_proj(reshape(x,n,n))
