function [obj,result_our] = MEA(X,groundtruth,knn,graph,lambda)

classnum = max(groundtruth);
viewnum = length(X); 
[num,~] = size(X);

for i = 1 :viewnum
    for  j = 1:num
         X{i}(j,:) = ( X{i}(j,:) - mean( X{i}(j,:) ) ) / std( X{i}(j,:) );
    end
end

for v = 1 :viewnum
    X{v} = X{v}';
end

options.NeighborMode = 'KNN';
options.k = knn; 
options.WeightMode = 'HeatKernel';
for v = 1:viewnum
    fea_v = X{1,v};
    if graph == 1
        A(v) = {constructW(fea_v',options)}; 
    else
        A(v) = {constructW_PKN(fea_v,knn)};
    end
end

dw = zeros(viewnum,1);

A_rep = zeros(num);
for v = 1:viewnum
   num = size(A{v},1);
   S10 = (A{v}+A{v}')/2;
   D10 = diag(sum(S10));
   L0 = D10 - S10;
   [F0, ~, ~] = eig1(full(L0), classnum+1, 0);
   F{v} = F0(:,1:classnum);
   A_sub{v} = F{v} * F{v}';
   A_rep = A_rep + A{v};
end

K0 = zeros(num,num);  
for v = 1:viewnum   
    K0 = K0 + A_sub{v};
end

W = orth(rand(num,classnum));

Itermax = 40;
obj = zeros(1,Itermax);
for iter = 1 : Itermax
    
    for v = 1:viewnum   
        dw(v) = 1/(2 * norm(W*W' - A{v}, 'fro'));
    end

    A0 = zeros(num,num);
    for v = 1 : viewnum
        A0 = A0 + dw(v)*A{v};
    end
    A_temp = 2*A0 + lambda * K0;
    [W, ~, ~] = eig1(A_temp, num, 1);
    W = W(:,2:classnum+1);
 
    % Calculate Obj
    res = 0;
    for v = 1 : viewnum
        res = res + norm(W*W' - A{v}, 'fro')^2;
    end
    obj(iter) = res - lambda*trace(W'*K0*W);

end

W = NormalizeFea(W,0);
result_our = kmeans(W, classnum, 'emptyaction', 'singleton', 'replicates', 100, 'display', 'off');

end