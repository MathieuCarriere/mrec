function [blk,At,b]=gh_constraints_equal(N,m)
%assumes N=m
tic
blk{1,1}='s'; blk{1,2}=N*m+1;

Auxt=spalloc( (N*m+1)*(N*m+2)/2 , 2*N+N*m*(m-1)+2 + N*(N-1)*m +2*N*m, 6*N*m*(m-1)); %first index probably wrong
b=zeros(2*N+N*m*(m-1)+2 + N*(N-1)*m +2*N*m,1);

%bottom entry
A=zeros(N*m+1,N*m+1);
A(N*m+1,N*m+1)=1;
Auxt(:,1)= svec(blk(1,:),A,1);
b(1)=1;

%trace constraint
Auxt(:,2)= svec(blk(1,:),eye(N*m+1),1);
b(2)=N+1;
idx=3;

for i=1:N
    sumoverj = zeros(N*m+1,N*m+1);
    for j=1:m
        sumoverj(N*m+1,(i-1)*m+j) = 1;
    end
    A=(sumoverj+sumoverj')/2;
    Auxt(:,idx)= svec(blk(1,:),A,1);
    b(idx)=1;
    idx=idx+1;
end

for j=1:m
    sumoveri = zeros(N*m+1,N*m+1);
    for i=1:N
        sumoveri(N*m+1,(i-1)*m+j) = 1;
    end
    A=(sumoveri+sumoveri')/2;
    Auxt(:,idx)= svec(blk(1,:),A,1);
    b(idx)=1;
    idx=idx+1;
end


for i=1:N
    sumoverj = zeros(N*m+1,N*m+1);
    for j=1:m
        sumoverj((i-1)*m+j,(i-1)*m+j) = 1;
    end
    A=(sumoverj+sumoverj')/2;
    Auxt(:,idx)= svec(blk(1,:),A,1);
    b(idx)=1;
    idx=idx+1;
end

for j=1:m
    sumoveri = zeros(N*m+1,N*m+1);
    for i=1:N
        sumoveri((i-1)*m+j,(i-1)*m+j) = 1;
    end
    A=(sumoveri+sumoveri')/2;
    Auxt(:,idx)= svec(blk(1,:),A,1);
    b(idx)=1;
    idx=idx+1;
end

for i=1:N
    for j=1:m
        for L=1:m
            if L~=j
                A= zeros(N*m+1,N*m+1);
                A((i-1)*m+j,(i-1)*m+L)=1;
                A=(A+A')/2;
                Auxt(:,idx)= svec(blk(1,:),A,1);
                b(idx)=0;
                idx=idx+1;
            end
        end
    end
end

for i=1:N
    for j=1:m
        for k=1:N
            if i~=k
                A= zeros(N*m+1,N*m+1);
                A((i-1)*m+j,(k-1)*m+j)=1;
                A=(A+A')/2;
                Auxt(:,idx)= svec(blk(1,:),A,1);
                b(idx)=0;
                idx=idx+1;
            end
        end
    end
end


[aux_A,idxs]=licols(sparse(Auxt),1e-5);
At{1}=sparse(aux_A);
b=b(idxs,:);
toc
end