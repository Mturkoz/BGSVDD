function[GP,GN,PosX,NegX] = Data_Seperation(TrData, TrLabels,kfold)
PosIdx = find(TrLabels == 1);
nP = length(PosIdx);
NegIdx = find(TrLabels == -1);
nN = length(NegIdx);
nSamp = nP + nN;
PosX = TrData(PosIdx,:);
NegX = TrData(NegIdx,:);
NP=size(PosX,1);
NN=size(NegX,1);
rng('default');
indices_P=crossvalind('Kfold',NP,kfold);
indices_N=crossvalind('Kfold',NN,kfold);

for i=1:kfold
    GP{i}= PosX(find(indices_P==i),:);
    GN{i}= NegX(find(indices_N==i),:);
end 
end