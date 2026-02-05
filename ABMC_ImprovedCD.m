function [F, label1, iter_num, obj0, obj] = ABMC_ImprovedCD(L,W,F0,lam)
% Adjustable Balanced Min-Max Cut for Graph Clustering
% Input:
% L is the Laplacian matrix n*n
% W is the similarity matrix n*n
% F0 is the initial label matrix of n samples n*c
% lam is the balance parameter
% Output:
% label1 is the label vector of n samples n*1
% F is the label matrix of n samples n*c
% iter_num is the number of iteration
% obj is the objective function value
%
% Coded by Qimin Liang

[n,~] = size(L);
[~,c] = size(F0);
last = 0;
iter_num = 0;
obj(1) = 0;
ftLf = zeros(1, c);
ftWf = zeros(1, c);
ftf = zeros(1, c);
ftL = zeros(c, n); 
ftW = zeros(c, n); 
F = F0;
[~, label1] = max(F, [], 2);
eps_safe = 1e-10;

%% compute Initial objective function value
for l = 1:c
    fl = F0(:, l);  
    Lfl = L * fl;
    Wfl = W * fl;
    numerator = fl' * Lfl;

    denominator = (fl' * Wfl + eps_safe)^lam;
    obj(1) = obj(1) + numerator / denominator;
end
obj0 = obj(1);
disp(['Objective function value before processing: ', num2str(obj(1))]);

while any(label1 ~= last)
    last = label1;
    %% Update F
    for i = 1:n
        p = label1(i) ;
        if sum(F(:, p)) == 1
            continue;
        end
        V1 = zeros(1, c);
        V2 = zeros(1, c);
        N1 = zeros(1, c);
        N2 = zeros(1, c);
        delta = zeros(1, c);
        for k = 1:c
            if k == p
                V1(k) = ftLf(k) - 2*ftL(k,i)+ L(i,i);
                N1(k) = ftWf(k) - 2*ftW(k,i);
                delta(k) = ftLf(k) / ((ftWf(k)+eps_safe)^lam) - V1(k) / ((N1(k)+eps_safe)^lam);
            else
                V2(k) = ftLf(k) + 2*ftL(k,i)+ L(i,i);
                N2(k) = ftWf(k) + 2*ftW(k,i);
                delta(k) = V2(k) / ((N2(k)+eps_safe)^lam) - ftLf(k) / ((ftWf(k)+eps_safe)^lam);
            end
        end
        [~,q] = min(delta);
        if p~=q
            ftLf(p) = V1(p);
            ftLf(q) = V2(q);
            ftWf(p) = N1(p);
            ftWf(q) = N2(q);
            ftL(p,:) = ftL(p,:) - L(i,:);
            ftL(q,:) = ftL(q,:) + L(i,:);
            ftW(p,:) = ftW(p,:) - W(i,:);
            ftW(q,:) = ftW(q,:) + W(i,:);
            label1(i) = q;
            F(i, :) = 0;
            F(i, q) = 1;  
        end

    end
    iter_num = iter_num+1;
    obj(iter_num) = 0;
    %% compute objective function value
    for l = 1:c
        obj(iter_num) = obj(iter_num) + ftLf(l) / ((ftWf(l)+eps_safe)^lam); %  objective function value
    end
end
for k = 1:c
    ftLf(k) = F(:, k)' * L * F(:, k);
    ftWf(k) = F(:, k)' * W * F(:, k); 
    ftf(k) = sum(F(:, k));
end
obj = obj(end);
end
