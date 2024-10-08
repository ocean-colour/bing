%perform the inversion for all possible values of the parameters.

function [N, adg_model, aph_model, apg_model, a_model, bbp_model, b_model]= IOP(L3, L4, Rrs, V, h, a_sea_water, bb_sea_water, wavelength, conv_criteria);

%generate all possible eigenvectors
%-----------------------------------------------------------------%
%generate vectors for absorption by NAP and CDOM:
S = [0.01:0.001:0.02];
for i = 1:length(S)
    a_dg(i, :) = exp(-S(i)*(wavelength-440));
end
%-----------------------------------------------------------------%
%generate vector(s) for absorption by phytoplankton:
%Sf = [0:0.1:1]; %use with the ciotti et al., approach and in Wang et al.,
%2005
Sf=1;
for m = 1:length(Sf)
    a_phi(m, :)=phyto_avg_abs(wavelength);
%    a_phi(m, :) = aphi(wavelength, Sf(m));
end
%-----------------------------------------------------------------%
%generate vectors for backscattering by particles:
Y = [0:0.2:2];
for n = 1:length(Y)
    bb_p(n, :) = (440./wavelength).^Y(n).*(V');
end
%-----------------------------------------------------------------%
%generate all possible combination of eigenvectors
k = 0;
for i = 1:length(S)
    for m = 1:length(Sf)
        for n = 1:length(Y)    
            k = k+1;
            B_matrix(:, :, k) = [S(i); Sf(m); Y(n)];
            D_matrix(:, :, k) = [a_dg(i, :); a_phi(m, :); bb_p(n, :)];
        end
    end
end
%-----------------------------------------------------------------%
%obtain linear solution by QR decomposition for every combination of igenvectors
for i = 1:length(S)*length(Sf)*length(Y)
    A=D_matrix(:, :, i)';
    b=h;
    [Q R]=qr(A);
    x = R\(R'\(A'*b));
    r = b - A*x;
    e = R\(R'\(A'*r));
    x = x + e;
    p_matrix(:, i)=x;
end
clear A b R Q x r e
%-----------------------------------------------------------------%
%for each solution obtain the IOP associated with that solution:
k=1;
for i=1:length(S)
    for m=1:length(Sf)
           for n=1:length(Y)
               A_dg(k,:)=a_dg(i,:)*p_matrix(1,k);
               A_phi(k,:)=a_phi(m,:)*p_matrix(2,k);
               B_bp(k,:)=bb_p(n,:)./V'*p_matrix(3,k);
               k=k+1;
           end
    end
end

for i=1:length(S)*length(Sf)*length(Y)
    a(i,:)=A_dg(i,:)+A_phi(i,:)+a_sea_water; %compute the total absorption
    b(i,:)=B_bp(i,:)+bb_sea_water; %compute total backscattering
end
%----------------------------------------------------------------%
%keep only realistic solutions:
B=find(p_matrix(1,:)>-0.005&p_matrix(2,:)>-0.005&p_matrix(3,:)>-0.0001);
%----------------------------------------------------------------%
%generate the Rrs based on the solutions:
for i=1:length(B)
    Rrs_model(i,:)=(0.0949)*(b(B(i),:)./(a(B(i),:) + b(B(i),:)))+(0.0794)*(b(B(i),:)./(a(B(i),:) + b(B(i),:))).^2;
end
%----------------------------------------------------------------%
%criteria for selection of solutions
o = 1;
for i = 1:length(B)
    M = find(abs(Rrs_model(i, :)-Rrs')<(conv_criteria*Rrs')); %match within convergence criteria at all wavelengths)
    if length(M) == length(Rrs)
        adg_model(o, :) = A_dg(B(i), :);
        aph_model(o, :) = A_phi(B(i), :);
        apg_model(o, :) = adg_model(o, :)+aph_model(o, :);
        bbp_model(o, :) = B_bp(B(i), :);
        a_model(o, :) = a(B(i), :);
        b_model(o, :) = b(B(i), :);
        rrs_model(o, :) = Rrs_model(i, :);
        eigen_adg(o) = B_matrix (1,1,B(i));
        eigen_aphi(o) = B_matrix (2,1,B(i));
        eigen_bbp(o) = B_matrix (3,1,B(i));
        N(o) = i;
        o = o+1;   
    end
end
%flag cases where no solution was found
if  o==1
    N=0;
    Rrs_model= [10 10 10 10 10];
    adg_model(1, :) = [10 10 10 10 10];
    aph_model(1, :) = [10 10 10 10 10];
    apg_model(1, :) = [10 10 10 10 10];
    bbp_model(1, :) = [10 10 10 10 10];
    a_model(1, :) = [10 10 10 10 10];
    b_model(1, :) = [10 10 10 10 10];
    rrs_model(1, :) = [10 10 10 10 10];
    eigen_adg(1) = [10];
    eigen_aphi(1) = [10];
    eigen_bbp(1) = [10];
end
%---------------------------------------------------------------%
