clear;


% % % Load Data
load Ecoli;

%% Training Data%%
TrData = [Ecoli(1:143,1:7);Ecoli(144:220,1:7)];
TrLabels=[ones(143,1);-ones(77,1)];
%
%%TEST DATA%%
TestData=Ecoli(221:end,1:7);
TestLabels =zeros(size(TestData,1),1);


PosIdx = find(TrLabels == 1);
nP = length(PosIdx);
NegIdx = find(TrLabels == -1);
nN = length(NegIdx);

sigma=3.39;
C1=25;
C2=0.25;
nu=0.05;
sigma11=1;
sigma22=2;
sigmaP=0.46;
V=0.775;

C12=1/(nu*nP);
C21=1/(nu*nN);


kfold=10;
[GP,GN,PosX,NegX] = Data_Seperation(TrData, TrLabels,kfold);



    for i=1:length(C1)
        
        for j=1:length(C2)
            
            for k=1:length(sigma)
                for m=1:kfold
                    
                    % Train Two-SVDD
                    XP=setdiff(PosX,GP{m},'rows');
                    XN=setdiff(NegX,GN{m},'rows');
                    TrData=[XP;XN];
                    TrLabels=[ones(size(XP,1),1);-ones(size(XN,1),1)];
                    
                    
                    [AlphaP, AlphaN, BetaP, BetaN, RsqP, RsqN] =BGSVDD_Train(sigma11,sigma22,sigmaP,V,TrData, TrLabels, C1(i),C2(j),C12,C21,sigma(k));
                    
                    % Test BGSVDD
                    TsData = [GP{m};GN{m};TestData];
                    True_Labels=[ones(size(GP{m},1),1);-ones(size(GN{m},1),1);TestLabels];
                    % Identify the labels of test data
                    [HatRsqP, HatRsqN,dLabels] = BGSVDD_Test(TrData, TrLabels, TsData, AlphaP, AlphaN, BetaP, BetaN, RsqP, RsqN, sigma(k));
                    % Calculate the Correctness
                    [F_SCORE] = Correctness(GP{m},GN{m},True_Labels,dLabels);                  
                    F_SCORE_COM(i,j,k,m)=F_SCORE;
                end
                Ave_F_SCORE_COM(i,j,k)=nanmean(F_SCORE_COM(i,j,k,:));    
                STD_F_SCORE_COM(i,j,k)=std(F_SCORE_COM(i,j,k,:)) ;    
            end
        end
    end
    


[F_MEASURE, idx2]=max(Ave_F_SCORE_COM(:));
a2=STD_F_SCORE_COM(:);
Standart_Dev_of_F_MEASURE=a2(idx2);
F_MEASURE
Standart_Dev_of_F_MEASURE





