function [F_SCORE] = Correctness(GP,GN,True_Labels,dLabels)



% True Positive
TP=size(find(True_Labels(1:size(GP,1)+size(GN,1))==dLabels(1:size(GP,1)+size(GN,1))),1);

%False Negative

FNegative=size(GP,1)+size(GN,1)-TP;

%True_Negative**
TN=size(find(True_Labels(size(GP,1)+size(GN,1)+1:size(True_Labels,1))==dLabels(size(GP,1)+size(GN,1)+1:size(True_Labels))),1);

%%%False Positive**
FalsePositive=size(True_Labels,1)-size(GP,1)-size(GN,1)-TN;

%Recall
Recall=TP/(TP+FNegative);
%Precision
Precision=TP/(TP+FalsePositive);

F_SCORE=2*(Precision*Recall)/(Precision+Recall);

end

