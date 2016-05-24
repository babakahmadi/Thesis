function [ Z, Beta ] = myKspca( X, Y, d, param )
%MYKSPCA [Z Beta] = myKspca(X, Y, d, param)
%{
    where
    Input:
        X: explanatory variable (pxn)
        Y: response variables (lxn)
        d: number of projection dimensions
    param:
        param.ktype_y : kernel type of the response variable
        param.kparam_y : kernel parameter of the response variable
        param.ktype_x : kernel type of the explanatory variable
        param.kparam_x : kernel parameter of the explanatory variable
    Output:
        Z: dimension reduced data (dxn)
        Beta: U = Phi(X) x Beta where U is the orthogonal projection matrix
%}
    param.ktype_y = 'delta'; % According to note 2
    
    n = size(Y,2);
    K = zeros(n);
    L = zeros(n);
    
    for i = 1 : n
        for j = 1 : n
            K(i,j) = kernel(param.ktype_x,X(:,i),X(:,j),param.kparam_x);
            L(i,j) = kernel(param.ktype_y,Y(:,i),Y(:,j),param.kparam_y);
        end
    end
    
    e = ones(n,1);
    I = eye(n);
    H = I - (1/n) * (e * e');
    
    Q = K*H*L*H*K;

    Beta = eigendec(Q,d,'LM');
    Z = Beta'*K;
end