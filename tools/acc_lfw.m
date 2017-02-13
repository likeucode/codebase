clear 
clc
path='./';
match=load([path 'match.txt']);
dismatch=load([path 'dismatch.txt']);
match=match';
dismatch=dismatch';
num_thr=1000;
num_fold=10;
thr=linspace(min([match dismatch]),max([match dismatch]),num_thr);
acc_thr=zeros(1,num_thr);
for i=1:num_thr
    acc_fold=zeros(1,num_fold);
    for j=1:num_fold
        match_fold=match(300*(j-1)+1:300*j);
        dismatch_fold=dismatch(300*(j-1)+1:300*j);
        positive=sum(match_fold>=thr(i))+sum(dismatch_fold<thr(i));
        negative=sum(match_fold<thr(i))+sum(dismatch_fold>=thr(i));
        acc_fold(j)=positive/(positive+negative);
    end
    acc_thr(i)=sum(acc_fold)/num_fold;
end
[best_acc,idx]=max(acc_thr);
disp (['Best Acc: ' num2str(best_acc)])
disp (['Best Threshold: ' num2str(thr(idx))])