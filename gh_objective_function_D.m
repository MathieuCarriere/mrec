function C=gh_objective_function_D(D1, D2)
N=size(D1, 2);
m=size(D2,2);

objmatrix = zeros(N*m+1,N*m+1);

for i=1:N
   for j=1:m
      for k=1:N
          for L=1:m
              objmatrix((i-1)*m+j,(k-1)*m+L) = abs(D1(i,k)-D2(j,L));
          end
      end
   end
end

C{1}=objmatrix;
end
