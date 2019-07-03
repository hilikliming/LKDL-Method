
%% Generating Training and Test Samples
X=randn(2,8000);
X= X-min(min(X));
X= X./max(max(X));
X=X-0.5;
for q=1:size(X,2)
   Y(1,q)= (X(1,q)>0 && X(2,q)<=0) || (X(1,q)<=0 && X(2,q)>0); 
end

% Partitioning samples into train and test
d.trainY=Y(1,1:end/2);
d.trainX=X(:,1:end/2);
d.testY=Y(1,end/2+1:end);
d.testX=X(:,end/2+1:end);

% Kernel Function Setup
sigma=.1;
kfnc = @(x,y) exp((-(repmat(sum(x.^2,1)',1,size(y,2))-2*(x'*y)+repmat(sum(y.^2,1),size(x,2),1))/(2*sigma))); %@(x,y) (x'*y);%  Gaussian Kernel

% Parameters for LKDL
smp_method='colnorm';
k=12;
c=4*k;

Xtrain=d.trainX;
Ytrain=d.trainY;
Xtest=d.testX;
Ytest=d.testY;

% Partitioning Train into H0 and H1 class samples
Xtrain0=Xtrain(:,Ytrain==0);
Xtrain1=Xtrain(:,Ytrain==1);

% Learning Parameters for baseline mapping to virtual sample space
[Ftrain,Ftest,XR,Sk,Vk] = LKDLPre(Xtrain,Xtest,kfnc,smp_method,c,k);
%[Ftrain0,Ftest0,XR0,~,~] = LKDLPre(Xtrain0,[],kfnc,smp_method,c,k);
%[Ftrain1,Ftest1,XR1,~,~] = LKDLPre(Xtrain1,[],kfnc,smp_method,c,k);

option.kernel = 'cust'; option.kernelfnc=kfnc;
% Creating C matrices for H0 and H1 samples
Ctrain0=computeKernelMatrix(Xtrain0,XR,option);
Ctrain1=computeKernelMatrix(Xtrain1,XR,option);

% Transforming H0 and H1 samples to F domain
Ftrain0=pinv(Sk)^(1/2)*Vk'*Ctrain0';
Ftrain1=pinv(Sk)^(1/2)*Vk'*Ctrain1';

% Finding first k/2 components of H0 and H1 subspaces in F space
[U0,~,~]=svd(Ftrain0,'econ');
[U1,~,~]=svd(Ftrain1,'econ');
Ftrain0=U0(:,1:k/2);
Ftrain1=U1(:,1:k/2);

% Generating Projection matrices for H0 and H1 class subspaces in F space
PH0=Ftrain0*(Ftrain0'*Ftrain0)^(-1)*Ftrain0';
PH1=Ftrain1*(Ftrain1'*Ftrain1)^(-1)*Ftrain1';

% generating H0 and H1 conditional statistics for test samples
h0stats=diag(Ftest'*PH0*Ftest);
h1stats=diag(Ftest'*PH1*Ftest);

% h0 and h1 stats in single matric (could form a LLR type stat with these
% for ROC generation if desired...
Yhat=[h0stats,h1stats];

% evaluating most likely classes based on subspace with better
% representaiton
[~,testYhat]=max(Yhat,[],2);
testY=Ytest';
figure;
% Plot confusion matrix for test samples
pltcnf(testY,logical(testYhat-1),1)

%% Plot all testing samples and red x's over those samples where misclassification was made.
testZ=Xtest;

figure; hold on;
plot(testZ(1,testY==0),testZ(2,testY==0),'b*');
plot(testZ(1,testY==1),testZ(2,testY==1),'go');

plot(testZ(1,testY~=testYhat-1),testZ(2,testY~=testYhat-1),'rx');
title('Linearized Kernel K-SVD SRC on XOR Problem');

num_correct=testY==testYhat-1;