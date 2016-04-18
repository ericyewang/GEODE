function y_m = GEODE_impute(U,SIGMAS,LAMBDA,MU,AA,Z,id_m,pos_m,IND,iscell)
% missing data imputation
d = size(LAMBDA,2);
SIG = (1./U-1)*SIGMAS;
SIG(setdiff(1:d,IND)) = zeros(d-length(IND),1);
if (iscell)
    y_m = cell(length(id_m),1);
    for i = 1:length(id_m)
        C_eta = bsxfun(@times,SIG,AA(:,:,i))/SIGMAS+eye(d);
        C_M = LAMBDA(pos_m{i},:)*(C_eta\bsxfun(@times,SIG,...
            LAMBDA(pos_m{i},:)')) + SIGMAS*eye(length(pos_m{i}));
        mu_hat = MU(pos_m{i}) + LAMBDA(pos_m{i},:)*(C_eta\(SIG.*Z(id_m(i),:)'))/SIGMAS;
        y_m{i} = mvnrnd(mu_hat,C_M);
    end
else
    y_m = [];
    for i = 1:length(id_m)
        C_eta = bsxfun(@times,SIG,AA(:,:,i))/SIGMAS+eye(d);
        C_M = LAMBDA(pos_m{i},:)*(C_eta\bsxfun(@times,SIG,...
            LAMBDA(pos_m{i},:)')) + SIGMAS*eye(length(pos_m{i}));
        mu_hat = MU(pos_m{i}) + LAMBDA(pos_m{i},:)*(C_eta\(SIG.*Z(id_m(i),:)'))/SIGMAS;
        tmp = mvnrnd(mu_hat,C_M);
        y_m = [y_m tmp];
    end
end