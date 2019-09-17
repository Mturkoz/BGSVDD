function [AlphaP, AlphaN, BetaP, BetaN, RsqP, RsqN] = BGSVDD_Train(sigma11,sigma22,sigmaP,V,TrData, TrLabels, C1,C2,C12,C21, sigma)

% BGSVDD Training with RBF kernel
%
% Inputs:
% TrData: N by d matrix (N samples, d dimensions)
% TrLabels: N by 1 vector {1, -1} (N samples)
% Nu1: Regularization parameter for the samples outside the sphere of its class
% Nu2: Regularization parameter for the samples inside the sphere of the other class
% Sigma: The width of RBF kernel
%
% Outputs:
% AlphaP: Lagrangian multiplier of positive class samples,
%         non-zero if the corresponding sample lies on or outside the sphere of positive class
% AlphaN: Lagrangian multiplier of negative class samples,
%         non-zero if the corresponding sample lies on or outside the sphere of negative class
% BetaP: Lagrangian multiplier of positive class samples,
%        non-zero if the corresponding sample lies on or inside the sphere of negative class
% BetaN: Lagrangian multiplier of negative class samples,
%        non-zero if the corresponding sample lies on or inside the sphere of positive class
% RsqP: Suared radius of positive class
% RsqN: Suared radius of negative class
%

% Initialization
RsqP = 0;
RsqN = 0;

PosIdx = find(TrLabels == 1);
nP = length(PosIdx);
NegIdx = find(TrLabels == -1);
nN = length(NegIdx);
nSamp = nP + nN;

PosX = TrData(PosIdx,:);
NegX = TrData(NegIdx,:);

% Construct the Kernel matrix
% Calculate K11
K11 = exp((-dist(PosX,PosX').^2)/(sigma^2));
% Calculate K12
K12 = exp((-dist(PosX,NegX').^2)/(sigma^2));
% Calculate K21
K21 = exp((-dist(NegX,PosX').^2)/(sigma^2));
% Calculate K22
K22 = exp((-dist(NegX,NegX').^2)/(sigma^2));
%Calculate whole data
% KK=exp((-dist(TrData,TrData').^2)/(sigma^2));

%%%Calculation of B matrices%%%
B1=diag(sum(K11'));
B2=diag(sum(K22'));
B12=diag(sum(K12'));
B21=diag(sum(K21'));

ones_K11=diag(ones(1,size(K11,1)));
ones_K22=diag(ones(1,size(K22,1)));

%%%%Calculate the Weights%%%%%


m_Alp_P=-(diag(B1)').^V;
m_Alp_N=-(diag(B2)').^V;
m_Beta_P=-(diag(B12)').^V;
m_Beta_N=-(diag(B21)').^V;


% Optimization process
% Preparing H and f for Quadratic Programming
% H for matlab QP routine
% H = 2*[K11 zeros(nP,nN) zeros(nP,nP) -K12 ; zeros(nN,nP) K22 -K21 zeros(nN,nN); ...
%     zeros(nP,nP) -K12 K11 zeros(nP,nN); -K21 zeros(nN,nN) zeros(nN,nP) K22];


% % % TEST NORMAL

H = 2*[nP*sigma11^-2*K11+sigmaP^-2*ones_K11 zeros(nP,nN) zeros(nP,nP) -nP*sigma11^-2*K12 ; zeros(nN,nP) nN*sigma22^-2*K22+sigmaP^-2*ones_K22 -nN*sigma22^-2*K21 zeros(nN,nN); ...
    zeros(nP,nP) -nN*sigma22^-2*K12 nN*sigma22^-2*K11 zeros(nP,nN); -nP*sigma11^-2*K21 zeros(nN,nN) zeros(nN,nP) nP*sigma11^-2*K22];


% Make sure D is positive definite:
i = -30;
while (pd_check(H + (10.0^i)*eye(size(H))) == 0)
    i = i+1;
end
i = i+5;
H = H + (10.0^i)*eye(size(H));

% % % TEST NORMAL
 f = [-2*sigma11^-2*diag(B1)-2*sigmaP^-2*m_Alp_P' ; -2*sigma22^-2*diag(B2)-2*sigmaP^-2*m_Alp_N' ;2*sigma22^-2*diag(B12)-2*sigmaP^-2*m_Beta_P' ;2*sigma11^-2*diag(B21)-2*sigmaP^-2*m_Beta_N'];

% Equality constraints on the variables
Aeq = [ones(1,nP) zeros(1,nN) zeros(1,nP) -ones(1,nN) ; zeros(1,nP) ones(1,nN) -ones(1,nP) zeros(1,nN)];
beq = [1 ; 1];
% Lower & Upper bounds for the variables
% Lower bound
LB = [zeros(1,nP) zeros(1,nN) zeros(1,nP) zeros(1,nN)]';
% Upper bound
UB = [ones(1,nP)*C1 ones(1,nN)*C2 ones(1,nP)*C12 ones(1,nN)*C21]';
% Initialization
rand('seed', sum(100*clock));
x0 = [ 0.5*rand(nSamp*2,1) ];

% Case 1: Original QP with Matlab Optimization Toolbox
opt = optimset; opt.LargeScale='off'; opt.Display='off'; opt.MaxIter=10000;
% Quadratic Optimization
LagM = quadprog(H,f,[],[],Aeq,beq,LB,UB,x0,opt);


% Set the value zero if it is too small
LagM(find(LagM < 10^-4)) = 0;
% Solutions
AlphaP = LagM(1:nP);
AlphaN = LagM(nP+1:nP+nN);
BetaP = LagM(nP+nN+1:2*nP+nN);
BetaN = LagM(2*nP+nN+1:end);

% Calculate the radius of each sphere
% Positive class
nbSVPIdx = find(AlphaP > 0 & AlphaP < C1-10^-5);
if length(nbSVPIdx) == 0
    nbSVPIdx = find(AlphaP > 0 & AlphaP < C1);
end
for i=1:length(nbSVPIdx)
    RsqP = RsqP + 1+AlphaP'*K11*AlphaP + BetaN'*K22*BetaN - 2*AlphaP'*K11(:,nbSVPIdx(i)) - 2*AlphaP'*K12*BetaN + 2*BetaN'*K21(:,nbSVPIdx(i));
end
RsqP = RsqP/length(nbSVPIdx);
% Negative class
nbSVNIdx = find(AlphaN > 0 & AlphaN < C2-10^-5);
if length(nbSVNIdx) == 0
    nbSVNIdx = find(AlphaN > 0 & AlphaN < C2);
end
for j=1:length(nbSVNIdx)
    RsqN = RsqN + 1+AlphaN'*K22*AlphaN + BetaP'*K11*BetaP - 2*AlphaN'*K22(:,nbSVNIdx(j)) - 2*AlphaN'*K21*BetaP + 2*BetaP'*K12(:,nbSVNIdx(j));
end
RsqN = RsqN/length(nbSVNIdx);





