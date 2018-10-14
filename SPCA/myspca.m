function [Z, U] = myspca(X, Y, d, param)
%MYSPCA [Z U] = myspca(X, Y, d, param)
%{
    where
    Input:
        X: explanatory variable (pxn)
        Y: response variables (lxn)
        d: dimension of effective subspaces
    param:
        param.ktype_y : kernel type of the response variable
        param.kparam_y : kernel parameter of the response variable
    Output:
        Z: dimension reduced data (dxn)
        U: orthogonal projection matrix (pxd)
%}
    param.ktype_y = 'delta'; % According to note 2

    n = size(Y,2);
    B = zeros(n);
    for i = 1 : n
        for j = 1 : n
            B(i,j) = kernel(param.ktype_y,Y(i),Y(j),param.kparam_y,[]);
        end
    end
    
    e = ones(n,1);
    I = eye(n);
    H = I - (1/n) * (e * e');
    
    Q = X * H * B * H * X';
    
    [U,~,~] = svd(Q);
    U = U(:,1:d);
    
    Z = U' * X * H; % Encoding
end

