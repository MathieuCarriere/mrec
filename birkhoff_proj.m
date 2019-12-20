function P=birkhoff_proj(X)
k=size(X,1);

f=reshape(-X, 1, k^2 );
A=zeros(2*k, k^2);
idx=1;
for i=1:k
   aux=zeros(k,k);
   aux(i,:)= ones(1,k);
   A(idx, :)=reshape(aux, 1, k^2);
   idx=idx+1;
   aux=zeros(k,k);
   aux(:,i)= ones(k,1);
   A(idx, :)=reshape(aux, 1, k^2);
   idx=idx+1;
end

P=reshape(linprog(f, [], [], A, ones(2*k,1), zeros(k^2,1), ones(k^2,1)), k,k);
end