function [X,Gamma,b,y,info,runhist]=gh_sdpnal_D(D1,D2)
N=size(D1,1);
m=size(D2,1);
C=gh_objective_function_D(D1,D2);


filename=sprintf('./constraints/constraints_%d_%d.mat', N,m);
if exist(filename, 'file') == 2
    load(filename,'blk','At','b');
else
    [blk, At,b]=gh_constraints_equal(N,m);
    save(filename, 'blk', 'At', 'b');
end


tic
OPTIONS.maxiter=50000;
OPTIONS.tol=1e-5;
[obj,X,s,y,S,Z,y2,v,info,runhist]=sdpnalplus(blk,At,C,b,0,[],[],[],[],OPTIONS);
toc
imagesc(cell2mat(X))
colormap('gray')
colormap(flipud(colormap));
X=cell2mat(X);
Gamma=C{1};
trace(Gamma*X)
end
