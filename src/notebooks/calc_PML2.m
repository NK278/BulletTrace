function [G2,G3,F1,F2,F3] = calc_PML2(IE,JE,npml)

IE = IE -1;
JE = JE - 1;
% Calculation of the PML parameters
G2 = ones(IE,JE);
G3 = ones(IE,JE);

F1 = zeros(IE,JE);
F2 = ones(IE,JE);
F3 = ones(IE,JE);

% Define parameters along the X dimension
gi2 = ones(1,IE);
gi3 = ones(1,IE);

fi1 = zeros(1,IE);
fi2 = ones(1,IE);
fi3 = ones(1,IE);

% Define parameters along the Y dimension
gj2 = ones(1,JE);
gj3 = ones(1,JE);

fj1 = zeros(1,JE);
fj2 = ones(1,JE);
fj3 = ones(1,JE);

% % Normal cells
for i = 0 : npml
    
    xnum = npml - i;
    xd = npml;
    xxn = xnum/xd;
    xn = 0.33*(xxn^3);
    
    gi2(i+1) = 1/(1+xn);
    gi2(IE-i) = 1/(1+xn);
    
    gi3(i+1) = (1-xn)/(1+xn);
    gi3(IE-i) = (1-xn)/(1+xn);
    
    xxn = (xnum-0.5)/xd;
    xn = 0.25*(xxn^3);
    
    fi1(i+1) = xn;
    fi1(IE-1-i) = xn;
    
    fi2(i+1) = 1/(1+xn);
    fi2(IE-1-i) = 1/(1+xn);

    fi3(i+1) = (1-xn)/(1+xn);
    fi3(IE-1-i) = (1-xn)/(1+xn);
    
end

for j= 0: npml
    
    xnum = npml-j;
    xd = npml;
    xxn = xnum/xd;
    xn = 0.33*(xxn^3);
    
    gj2(j+1) = 1/(1+xn);
    gj2(JE-j) = 1/(1+xn);
    
    gj3(j+1) = (1-xn)/(1+xn);
    gj3(JE-j) = (1-xn)/(1+xn);
  
    xxn = (xnum-0.5)/xd;
    xn = 0.25*(xxn^3);
    
    fj1(j+1)=xn;
    fj1(JE-1-j)= xn;
    
    fj2(j+1)= 1/(1+xn);
    fj2(JE-1-j)=1/(1+xn);
   
    fj3(j+1)=(1-xn)/(1+xn);
    fj3(JE-1-j)=(1-xn)/(1+xn);
    
end

G2 = gi2.'*gj2;
G3 = gi3.'*gj3;
F1 = fi1.'*fj1;
F2 = fi2.'*fj2;
F3 = fi3.'*fj3;







