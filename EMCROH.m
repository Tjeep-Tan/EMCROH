
clear all;
clear;


opts.dirs.data = '../data';
opts.unsupervised = 0;
opts.nbits = 32;
normalizeX = 1;


%  DS = Datasets.places(opts, normalizeX);
%   DS = Datasets.cifar(opts, normalizeX);
  DS = Datasets.mnist(opts, normalizeX);
% DS = Datasets.nuswide(opts, normalizeX);


trainCNN = DS.Xtrain;  % n x  d 
testCNN = DS.Xtest;  
trainLabels = DS.Ytrain;  %  n x d
testLabels = DS.Ytest;

% mapped into a sphere space
test = testCNN ./ sqrt(sum(testCNN .* testCNN, 2));  
%test = testCNN;
testLabel = testLabels;  % n x 1
train = trainCNN ./ sqrt(sum(trainCNN .* trainCNN, 2));   
%train = trainCNN;
trainLabel = trainLabels; % n x 1
clear testCNN trainCNN testLabels trainLabels

[Ntrain, Dtrain] = size(train);
[Ntest, Dtest] = size(test);

test = test';
train = train';

W_t = randn(Dtest, opts.nbits);  
W_t = W_t ./ repmat(diag(sqrt(W_t' * W_t))', Dtest, 1);

%%%%%%%%%%%% Four parameters depicted in the paper %%%%%%%%%%%%%%%%
lambda = 2;   
sigma = 1.8;    
etad = 0.2;     
etas = 1.2;     
lambda1 = 12;   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_t = 200;     % training size at each stage      
training_size =6000;   % total training instances 

Xs_t = [];
Bs_t = [];
ls_t = [];

Be_t = [];
Xe_t = [];
le_t = [];
F = [];
G = [];
S_t = [];

Xe_t = train(:, 1 : n_t);
tmp = W_t' * Xe_t;
tmp(tmp >= 0) = 1;
tmp(tmp < 0) = -1;
Be_t = tmp; 
le_t = trainLabel(1 : n_t); 

for t = n_t:n_t:training_size 
    if t + n_t > Ntrain  
        break
    end

    Xs_t = train(:, t - n_t + 1 : t); 
    tmp = W_t' * Xs_t;
    tmp(tmp >= 0) = 1;
    tmp(tmp < 0) = -1;
    Bs_t = tmp;  
    
    ls_t = trainLabel(t - n_t + 1 : t);
    S_t = single(ls_t == le_t');   
    for i = 1:n_t
        if sum(S_t(i,:)) ~= 0
            ind = find(S_t(i,:) ~=0);
            Be_t(:, i) = Bs_t(:, ind(1));
        end
    end
    S_t(S_t == 0) = -etad;
    S_t(S_t == 1) = etas;
    S_t =2 * S_t * opts.nbits;  

    F = Bs_t' * Be_t;
    F(F >= 0) = 1;
    F(F < 0) = -1;
    
% update G 
    G = F - S_t;    
    G(G >= 0) = 1;
    G(G < 0) = -1; 
% update F 
    F = G + S_t + (Bs_t' * Be_t)/200; % 200

%   update Bs
    Bs_t = updateColumnBs_t(Bs_t', Be_t', S_t, F, opts.nbits, lambda1);   
    
    tmp = W_t' * Xs_t;
    tmp(tmp >= 0) = 1;
    tmp(tmp < 0) = -1;
    Bs_t = Bs_t + tmp;
    Bs_t(Bs_t >= 0) = 1;
    Bs_t(Bs_t < 0) = -1;

% update Be
    Be_t = (Be_t + Bs_t * S_t)/2;
    Be_t(Be_t >= 0) = 1;
    Be_t(Be_t < 0) = -1;


% update W 
    I = eye(Dtrain);


    W_t = sigma * (sigma * (Xs_t * Xs_t'+ Xe_t * Xe_t')+ lambda * I) \ ( Xs_t * Bs_t'+ Xe_t * Be_t');
   
    Xe_t = [Xe_t, Xs_t];
    Be_t = [Be_t, Bs_t]; 
    le_t = [le_t; ls_t]; 
end



Htrain = single(W_t' * train > 0);
Htest = single(W_t' * test > 0);

Aff = affinity([], [], trainLabel, testLabel, opts);

opts.metric = 'mAP';
res = evaluate(Htrain', Htest', opts, Aff);


function U = updateColumnBs_t(U, V, S,F, bit, lambda)
m = 0.1;
n = size(U, 1);
    TX = lambda * S * V / bit;
    UV = U*V';
    AX =(2*(S-UV)*V*lambda^2) /((1 + exp(norm(S-UV)))*(bit^2));
    Be_F = F*V;
for k = 1: bit
    B = TX(:,k) - AX(:,k);
    p = lambda *(B+Be_F(:,k))/bit + m * lambda^2 * U(:, k) / (4 * bit^2);
    U_opt = ones(n, 1);
    U_opt(p <= 0) = -1;
    U(:, k) = U_opt;
end
   U = U';
end

function V = updateColumnBe_t(V, U, S,F, bit, lambda)
m = 0.1;
n = size(U, 1);
   TX = lambda * S * U / bit;
   VU = V*U';
   AX =(2*(S-VU)*U*lambda^2) /((1 + exp(norm(S-VU)))*(bit^2));
    BsF = F'*U;
for k = 1: bit
    B = TX(:,k) - AX(:,k);
    p = lambda *(B+BsF(:,k))/bit + m * lambda^2 * V(:, k) / (4 * bit^2);
    V_opt = ones(n, 1);
    V_opt(p <= 0) = -1;
    V_opt(p > 0) = 1;
    V(:, k) = V_opt;
end
    V = V';
end

