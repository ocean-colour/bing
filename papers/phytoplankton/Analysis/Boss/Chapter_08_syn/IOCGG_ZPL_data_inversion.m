%This is the driver program to invert Rrs into IOP + uncertainties in inversion products
%This an implemention of the strategy suggeted in the Wang, Boss and Roesler, 2005,
%Applied Optics paper.
%Here it is used to invert in IOCGG in-situ data sets. With slight
%modifications it can be applied to other data sets

%calls the following m.files:
%IOP_of_sw.m, IOP_inversion.m, v.m, h_Array.m, and reg_stats.m

%Emmanuel Boss, U Maine, 2006

clear all
close all

%convergence criteria for selection of solutions. e.g. if 0.1 only solution
%that are within 10% of Rrs at all wavelengths will be considered. 
conv_criteria=0.1;

%Read input data
%seawater IOPs used by ZP Lee:
A=load('wl_aw_bbw.txt');
wavelength=[410	440	490	510	550];
for i=1:length(wavelength)
    I(i)=find(A(:,1)==wavelength(i));
end
a_sea_water=A(I,2)';
bb_sea_water=A(I,3)';

%Other validation data
B=load('a_dg.txt'); %absorption by NAP+CDOM
C=load('a_ph.txt'); %absorption by phytoplankton
D=load('bb_total.txt'); %particulate backscattering
E=load('a_total.txt');  %total absorption (not including water)
G=load('a_g.txt'); %CDOM

%Rrs to invert:
F=load('r_rs30degree.txt'); 

a_phi=C(:,I);
a_dg=B(:,I)+G(:,I);
a_pg=E(:,I);
bb_tot=D(:,I);
Rrs=F(:,I);

%paramters of model relating Rrs to IOP
L3 = 0.0949;
L4 = 0.0794;

for i=1:500
    i
    %Setting up the linear solver:
    [V] = v(L3, L4, Rrs(i,:)');
    [h] = Array_h(a_sea_water, bb_sea_water, V);

    %proceed with inversion
    [N, adg_model, aph_model, apg_model, a_model, bbp_model, b_model]= IOP_inversion(L3, L4, Rrs(i,:)', V, h, a_sea_water, bb_sea_water, wavelength, conv_criteria);

    if N>0
    adg_model_median(i,:)=mean(adg_model);
    adg_model_95(i,:)=prctile(adg_model,95);
    adg_model_5(i,:)=prctile(adg_model,5);
    aph_model_median(i,:)=mean(aph_model);
    aph_model_95(i,:)=prctile(aph_model,95);
    aph_model_5(i,:)=prctile(aph_model,5);
    a_model_median(i,:)=mean(a_model);
    a_model_95(i,:)=prctile(a_model,95);
    a_model_5(i,:)=prctile(a_model,5);
    bbp_model_median(i,:)=mean(bbp_model);
    bbp_model_95(i,:)=prctile(bbp_model,95);
    bbp_model_5(i,:)=prctile(bbp_model,5);
    bb_model_median(i,:)=mean(b_model);
    bb_model_95(i,:)=prctile(b_model,95);
    bb_model_5(i,:)=prctile(b_model,5);
    nn(i)=length(N);
    else
    adg_model_median(i,:)=[10 10 10 10 10];
    adg_model_95(i,:)=[10 10 10 10 10];
    adg_model_5(i,:)=[10 10 10 10 10];
    aph_model_median(i,:)=[10 10 10 10 10];
    aph_model_95(i,:)=[10 10 10 10 10];
    aph_model_5(i,:)=[10 10 10 10 10];
    a_model_median(i,:)=[10 10 10 10 10];
    a_model_95(i,:)=[10 10 10 10 10];
    a_model_5(i,:)=[10 10 10 10 10];
    bbp_model_median(i,:)=[10 10 10 10 10];
    bbp_model_95(i,:)=[10 10 10 10 10];
    bbp_model_5(i,:)=[10 10 10 10 10];
    bb_model_median(i,:)=[10 10 10 10 10];
    bb_model_95(i,:)=[10 10 10 10 10];
    bb_model_5(i,:)=[10 10 10 10 10];
    nn(i)=length(N);
    end
end
I=find(nn>1);
figure(1)
subplot(2,2,1)
loglog(a_pg(I,2),a_model_median(I,2),'o');
hold on
for i=1:length(I)
    plot([a_pg(I(i),2),a_pg(I(i),2)],[a_model_5(I(i),2),a_model_95(I(i),2)],'b-')
end
hold on
plot([0.01 5],[0.01 5],'k-')
axis([0.01 5 0.01 5])
xlabel('Known a_{pg}(440)')
ylabel('derived a_{pg}(440)')
hold off

subplot(2,2,2)
loglog(a_phi(I,2),aph_model_median(I,2),'o');
hold on
for i=1:length(I)
    plot([a_phi(I(i),2),a_phi(I(i),2)],[aph_model_5(I(i),2),aph_model_95(I(i),2)],'b-')
end
hold on
plot([0.003 1],[0.003 1],'k-')
axis([0.003 1 0.003 1])
xlabel('Known a_{ph}(440)')
ylabel('derived a_{ph}(440)')
hold off

subplot(2,2,3)
loglog(a_dg(I,2),adg_model_median(I,2),'o');
hold on
for i=1:length(I)
    plot([a_dg(I(i),2),a_dg(I(i),2)],[adg_model_5(I(i),2),adg_model_95(I(i),2)],'b-')
end
hold on
plot([0.0005 3],[0.0005 3],'k-')
axis([0.0005 3 0.0005 3])
xlabel('Known a_{dg}(440)')
ylabel('derived a_{dg}(440)')
hold off

subplot(2,2,4)
loglog(bb_tot(I,2),bb_model_median(I,2),'o');
hold on
for i=1:length(I)
    plot([bb_tot(I(i),2),bb_tot(I(i),2)],[bb_model_5(I(i),2),bb_model_95(I(i),2)],'b-')
end
hold on
plot([0.002 0.2],[0.002 0.2],'k-')
axis([0.002 0.2 0.002 0.2])
xlabel('Known b_{b}(440)')
ylabel('derived b_{b}(440)')
hold off
I=find(aph_model_median(:,2)>0 & adg_model_median(:,2)>0 & bb_model_median(:,2)>0 & nn(:)>1);
[N, intercept_t1, intercept_t2, slope_t1, slope_t2, rsq, rms1, bias]=reg_stats(a_model_median(I,2),a_pg(I,2),10);
[N, intercept_t1, intercept_t2, slope_t1, slope_t2, rsq, rms1, bias]=reg_stats(aph_model_median(I,2),a_phi(I,2),10);
[N, intercept_t1, intercept_t2, slope_t1, slope_t2, rsq, rms1, bias]=reg_stats(adg_model_median(I,2),a_dg(I,2),10);
[N, intercept_t1, intercept_t2, slope_t1, slope_t2, rsq, rms1, bias]=reg_stats(bb_model_median(I,2),bb_tot(I,2),10);
