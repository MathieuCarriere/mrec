function [Aeq,beq]=gh_constraints_cplex(n)

N=n^2;
Aeq=zeros(2*n,N);
beq=ones(2*n,1);


for j=1:n
    A=zeros(n,n);
    A(j,:)=ones(1,n);
    Aeq(j,:)=reshape(A,1, N);
end

for j=1:n
    A=zeros(n,n);
    A(:,j)=ones(n,1);
    Aeq(j+n,:)=reshape(A,N,1);
end

end
