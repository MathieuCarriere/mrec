addpath('./cplex/')

A1 = cell2mat([D1{:}]);
A2 = cell2mat([D2{:}]);
sz1 = size(A1);
sz2 = size(A2);
s1 = sqrt(sz1(2));
s2 = sqrt(sz2(2));

[X,val] = gh_cplex(double(reshape(A1,s1,s1)), double(reshape(A2,s2,s2)), maxtime);
