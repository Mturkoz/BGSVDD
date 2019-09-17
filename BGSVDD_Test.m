function [HatRsqP, HatRsqN,dLabels] = BGSVDD_Test(TrData, TrLabels, TsData, AlphaP, AlphaN, BetaP, BetaN, RsqP, RsqN, sigma)
% Two-Class SVDD Test with RBF kernel
%
% Inputs:
% TrData: N by d matrix (N samples, d dimensions)
% TrLabels: N by 1 vector {1, -1} (N samples)
% TsData: M by d matrix (M samples, d dimensions)
% AlphaP: Lagrangian multiplier of positive class samples,
%              non-zero if the corresponding sample lies on or outside the sphere of positive class
% AlphaN: Lagrangian multiplier of negative class samples,
%              non-zero if the corresponding sample lies on or outside the sphere of negative class
% BetaP: Lagrangian multiplier of positive class samples,
%            non-zero if the corresponding sample lies on or inside the sphere of negative class
% BetaN: Lagrangian multiplier of negative class samples,
%             non-zero if the corresponding sample lies on or inside the sphere of positive class
% Sigma: The width of RBF kernel
% RsqP: Suared radius of positive class
% RsqN: Suared radius of negative class
%
% Outputs:
% HatRsqP: squared distance to the center of positive sphere
% HatRsqN: squared distance to the center of negative sphere
% dLabels: predicted labels by absolute distance
% if d-R1 < d-R2
%     assign x to Class 1
% else
%     assign x to Class 2
% rLabels: predicted labels by relative distance
% if d/R1 < d/R2
%     assign x to Class 1
% else
%     assign x to Class 2

% Initialization
[nSamp, nFeat] = size(TsData);
HatRsqP = zeros(nSamp,1);
HatRsqN = zeros(nSamp,1);
dLabels = zeros(nSamp,1);
rLabels = zeros(nSamp,1);

% Training data separation
TrPosIdx = find(TrLabels == 1);
TrNegIdx = find(TrLabels == -1);
TrPosData = TrData(TrPosIdx,:);
TrNegData = TrData(TrNegIdx,:);

% Support Vector Extraction
% Indexing
SVAPIdx = find(AlphaP > 0);
SVANIdx = find(AlphaN > 0);
SVBPIdx = find(BetaP > 0);
SVBNIdx = find(BetaN > 0);
% SVs
SVAP = TrPosData(SVAPIdx,:);
SVAN = TrNegData(SVANIdx,:);
SVBP = TrPosData(SVBPIdx,:);
SVBN = TrNegData(SVBNIdx,:);
% Corresponding multipliers
NZAlphaP = AlphaP(SVAPIdx);
NZAlphaN = AlphaN(SVANIdx);
NZBetaP = BetaP(SVBPIdx);
NZBetaN = BetaN(SVBNIdx);

% Assign HatRsqP
T1 = NZAlphaP'*exp(-(dist(SVAP,SVAP').^2)/(sigma^2))*NZAlphaP;
T2 = NZBetaN'*exp(-(dist(SVBN,SVBN').^2)/(sigma^2))*NZBetaN;
T3 = NZAlphaP'*exp(-(dist(SVAP,SVBN').^2)/(sigma^2))*NZBetaN;
for i=1:nSamp
    HatRsqP(i,1) = 1 + T1 + T2 - 2*T3 - 2*NZAlphaP'*exp(-(dist(SVAP,TsData(i,:)').^2)/(sigma^2)) + 2*NZBetaN'*exp(-(dist(SVBN,TsData(i,:)').^2)/(sigma^2));
end
% Assign HatRsqN
T4 = NZAlphaN'*exp(-(dist(SVAN,SVAN').^2)/(sigma^2))*NZAlphaN;
T5 = NZBetaP'*exp(-(dist(SVBP,SVBP').^2)/(sigma^2))*NZBetaP;
T6 = NZAlphaN'*exp(-(dist(SVAN,SVBP').^2)/(sigma^2))*NZBetaP;
for j=1:nSamp
    HatRsqN(j,1) = 1 + T4 + T5 - 2*T6 - 2*NZAlphaN'*exp(-(dist(SVAN,TsData(j,:)').^2)/(sigma^2)) + 2*NZBetaP'*exp(-(dist(SVBP,TsData(j,:)').^2)/(sigma^2));
end

%%Find the observations which are outside of both hypersphers%%

for i=1:nSamp
    
    if HatRsqP(i)<RsqP & HatRsqN(i)>RsqN
        dLabels(i)=1;
    elseif HatRsqN(i)<RsqN & HatRsqP(i)>RsqP 
        dLabels(i)=-1;
    elseif HatRsqP(i)<RsqP & HatRsqP(i)<HatRsqN(i)
        dLabels(i)=1;
    elseif HatRsqP(i)<RsqP & HatRsqN(i)<HatRsqP(i)
        dLabels(i)=-1;
    elseif HatRsqP(i)>RsqP  & HatRsqN(i)>RsqN
        dLabels(i)=0;
    end
end






end
    
    
