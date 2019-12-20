function C=gh_objective_cplex(D1, D2)
N=size(D1, 2);
m=size(D2,2);

objmatrix = zeros(N*m,N*m);

for i=1:N
   for j=1:m
      for k=1:N
          for L=1:m
              objmatrix((i-1)*m+j,(k-1)*m+L) = abs(D1(i,k)-D2(j,L));
          end
      end
   end
end

C=objmatrix;
end
