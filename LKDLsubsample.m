function [XR] = LKDLsubsample(X,smp_method,kfnc,c)

option.kernel = 'cust'; option.kernelfnc=kfnc;
K=computeKernelMatrix(X,X,option);

switch smp_method
    case 'colnorm'
        norms=vecnorm(K);
        [~,ind]=sort(abs(norms),'descend');
        XR=X(:,ind(1:c));
    case 'diagonal'
        diags=diag(K);
        [~,ind]=sort(diags,'descend');
        XR=X(:,ind(1:c));
    otherwise
        ind=randperm(size(X,2),c);
        XR=X(:,ind);
end
end

