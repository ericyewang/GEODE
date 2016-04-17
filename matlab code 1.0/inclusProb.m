function inclu = inclusProb(d,IND,ADA,T,varargin)
% Compute the inclusion probability of each dimension
%% arguments
numvarargs = length(varargin);
if numvarargs > 2
    error('inclusProb:TooManyInputs', ...
        'requires at most 2 optional inputs');
end
optargs = {1 0.5};
optargs(1:numvarargs) = varargin;
[verbose, threshold] = optargs{:};
%% Compute Inclusion Chance
adanum = sum(ADA);
inclu = zeros(d,1);
TT = 1:T;
for iter = TT(logical(ADA))
    for j = 1:d
        if find(IND{iter}==j)
        inclu(j) = inclu(j) + 1;
        end
    end
end
inclu = inclu/adanum;
%% Plot
if verbose
bar(inclu);
hold on
plot(get(gca,'xlim'), [threshold threshold],'--');
hold off
title('Inclusion Chance for all Dimensions');
end