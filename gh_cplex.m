function [X,val]=gh_cplex(X1,X2, maxtime)

n=size(X1,1);
D1=zeros(n,n);
D2=zeros(n,n);
for i=1:n
    for j=1:n
        D1(i,j)=norm(X1(i,:)-X1(j,:),2);
        D2(i,j)=norm(X2(i,:)-X2(j,:),2);
    end
end
N=n*n;
H=gh_objective_cplex(D1,D2);

[Aeq,beq]=gh_constraints_cplex(n);

tic

for i=1:N
    ctype(i)='B';
end
options = cplexoptimset;
options.Display = 'off';
options.MaxTime = maxtime;
f=zeros(N,1);

[X,val]=cplexmiqp(H,f, [],[],Aeq, beq,[],[],[], [], [], ctype,[], options);
toc
imagesc(reshape(X,n,n))
end
