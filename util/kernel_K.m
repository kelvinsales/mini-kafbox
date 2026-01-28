function [M mat1 mat2 P]=kernel_K(x,y,tipo,gamma=0,bias=0)
%
%        Funcao de KERNEL a ser utilizada
%                    17/08/2018
%
% TIPOS:
%  (1) 'LIN'  = Linear
%  (2) 'GAUS' = Gaussiano
%
%=====================================================================
%

kernel = tipo;
##x=x';
##y=y';
Cx=size(x,1);
Cy=size(y,1);

C1=gamma;
C2=bias;

M=zeros(Cx,Cy);
K=zeros(Cx,Cy);
%
switch kernel

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-----------------------------------------------------------
% Linear
  case 'LIN'
    %
    if(size(x,1)>1)
      M = x*x'+C1;
    else
      M = x'*x+C1;
    end

    if (!isempty(y)|| y!=0)
      M=x*y'+C1;
    endif
%-----------------------------------------------------------
% Gaussiano
  case 'GAUS'
    %
    norms1 = sum(x.^2,2);
    norms2 = sum(y.^2,2);

    mat1 = repmat(norms1,1,Cy);
    mat2 = repmat(norms2',Cx,1);
    P=(x*y');
    dist2 = mat1 + mat2 - 2*P; % full distance matrix
    M = exp(-dist2*C1);
##    K = pdist2(x,y).^2;
##    M = exp(-K*C1);
##    if Cx == Cy
##        M = (exp(-sum((x-y).^2)*C1))';
##    elseif Cx == 1
##        M = (exp(-sum((x*ones(1,Cy)-y).^2)*C1))';
##    elseif Cy == 1
##        M = (exp(-sum((x-y*ones(1,Cx)).^2)*C1))';
##    else
##        printf('error dimension--\n')
##    end
##    M=exp(-dist2*C1);
##    if isempty(y)
##      dist2=(x-x').^2;
##    else
##      s1=diag(x*x');
##      s2=diag(y*y');
##      dist2=s1+s2'-2*x*y';
##    endif
##    M = exp(-dist2*C1);
##    M(M<10.^-10)=0;

%-----------------------------------------------------------
% Polyan
  case  'POLY'
    %
    M = (x*y'+C2).^C1;

%-----------------------------------------------------------
% Tangente hiperpolica
  case  'MLP'
    if(Cx==Cy)
      M = tanh(C1*x*y'+C2);
    else
      M = tanh(C1*x*y'+C2);
    end
##    M = tanh(C1*x*y'+C2);
end
