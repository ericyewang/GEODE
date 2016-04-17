function [pred_summary,pred_sample] = GEODE_predict(y,Lambda,mu,InD,u,sigmaS)
%% Preparation
[N,D] = size(y); d = size(Lambda,2);
T = length(sigmaS);
%% Extract Missing Positions
id_m = 1:N;
pos_m = cell(N,1);
id_N = [];
id_position = [];
for i = 1:N
    pos_m{i} = find(isnan(y(i,:)));
    id_N = [id_N i*ones(1,length(pos_m{i}))];
    id_position = [id_position pos_m{i}];
end
pred_sample = zeros(length(id_N),T);
pred_summary = zeros(length(id_N),7);
pred_summary(:,1) = id_N;
pred_summary(:,2) = id_position;
%% Store Sufficient Statistics
fun2 = @(n) (y(n,~isnan(y(n,:)))-mu(~isnan(y(n,:)))')*Lambda(~isnan(y(n,:)),:);
Z = arrayfun(fun2,1:N, 'UniformOutput',0);
Z = cat(1,Z{:});
AA = zeros(d,d,length(id_m));
for k = 1:length(id_m)
    AA(:,:,k) = Lambda(setdiff(1:D,pos_m{i}),:)'*Lambda(setdiff(1:D,pos_m{i}),:);
end
%% Sample from the Predictive Posterior Distribution
for iter = 1:T
    pred_sample(:,iter) = GEODE_impute(u(:,iter),sigmaS(iter),Lambda,mu,AA,Z,id_m,pos_m,InD,0);
end
pred_summary(:,3) = mean(pred_sample,2); % posterior predictive mean
pred_summary(:,4) = median(pred_sample,2); % posterior predictive median
pred_summary(:,5) = quantile(pred_sample,0.025,2); % posterior predictive 2.5% quantile
pred_summary(:,6) = quantile(pred_sample,0.975,2); % posterior predictive 97.5% quantile
pred_summary(:,7) = std(pred_sample'); % posterior predictive standard deviation