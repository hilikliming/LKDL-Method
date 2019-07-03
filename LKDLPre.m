function [Ftrain,Ftest,XR,Sk,Vk] = LKDLPre(Xtrain,Xtest,kfnc,smp_method,c,k)
% Pre-processing required to use LKDL method.

option.kernel = 'cust'; option.kernelfnc=kfnc;

XR=LKDLsubsample(Xtrain,smp_method,kfnc,c);
Ctrain=computeKernelMatrix(Xtrain,XR,option);
W=computeKernelMatrix(XR,XR,option);
[V,S,~]=svd(W,'econ');
Sk=S(1:k,1:k);
Vk=V(:,1:k);
Wk= Vk*Sk*Vk';
Ftrain=pinv(Sk)^(1/2)*Vk'*Ctrain';
if ~isempty(Xtest)
    Ctest=computeKernelMatrix(Xtest,XR,option);
    Ftest=pinv(Sk)^(1/2)*Vk'*Ctest';
else
    Ftest=[];
end



end

