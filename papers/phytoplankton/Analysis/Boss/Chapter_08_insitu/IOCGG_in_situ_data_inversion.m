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

%Read input data
A=load('insitu_data2_reduced.txt');
Rrs=A(:,2:6);  %Rrs for inversion

chl=A(:,1);  %chlorophyll for comparison
a_phi=A(:,7:11);  %phytoplankton absorption for comparison
a_dg=A(:,12:16);  %absorption by CDOM+NAP for comparison
a_pg=A(:,17:21);  %absorption of total particles for comparison
wavelength=[412	443	490	510	555];

%user inputs:
%Temperature and salinity of water under consideration [degree C] (to compute water IOPs)
Temp=20; Salt = 30;

%compute IOPs of seawater
[a_sea_water,bb_sea_water]=IOP_of_sw(wavelength,Temp,Salt);

%convergence criteria for selection of solutions. e.g. if 0.1 only solution
%that are within 10% of Rrs at all wavelengths will be considered. 
conv_criteria=0.1;


%paramters of model relating Rrs to IOP (here based on GSM)
L3 = 0.0949;
L4 = 0.0794;

%Loop where all the spectra are inverted
for i=1:length(chl)
    %Setting up the linear solver:
    i
    [V] = v(L3, L4, Rrs(i,:)');
    [h] = Array_h(a_sea_water, bb_sea_water, V);

    %proceed with inversion
    [N, adg_model, aph_model, apg_model, a_model, bbp_model, b_model]= IOP_inversion(L3, L4, Rrs(i,:)', V, h, a_sea_water, bb_sea_water, wavelength, conv_criteria);
    
    %calculate the statistics of all the possible solutions found:
    if N>0
    adg_model_median(i,:)=median(adg_model);
    adg_model_95(i,:)=prctile(adg_model,95);
    adg_model_5(i,:)=prctile(adg_model,5);
    aph_model_median(i,:)=median(aph_model);
    aph_model_95(i,:)=prctile(aph_model,95);
    aph_model_5(i,:)=prctile(aph_model,5);
    a_model_median(i,:)=median(a_model);
    a_model_95(i,:)=prctile(a_model,95);
    a_model_5(i,:)=prctile(a_model,5);
    nn(i)=length(N);
    else     %flag cases where no solution was found
    adg_model_median(i,:)=[10 10 10 10 10];
    adg_model_95(i,:)=[10 10 10 10 10];
    adg_model_5(i,:)=[10 10 10 10 10];
    aph_model_median(i,:)=[10 10 10 10 10];
    aph_model_95(i,:)=[10 10 10 10 10];
    aph_model_5(i,:)=[10 10 10 10 10];
    a_model_median(i,:)=[10 10 10 10 10];
    a_model_95(i,:)=[10 10 10 10 10];
    a_model_5(i,:)=[10 10 10 10 10];
    nn(i)=length(N);
    end
end

%plot data for all the retreived solutions
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


%calculate the parametric and nonparametric statistics of all the median solution:
[N, intercept_t1, intercept_t2, slope_t1, slope_t2, rsq, rms1, bias]=reg_stats(a_model_median(I,2),a_pg(I,2),10);
[N, intercept_t1, intercept_t2, slope_t1, slope_t2, rsq, rms1, bias]=reg_stats(aph_model_median(I,2),a_phi(I,2),10);
I=find(adg_model_median(:,2)>0 & nn(:)>1);
[N, intercept_t1, intercept_t2, slope_t1, slope_t2, rsq, rms1, bias]=reg_stats(adg_model_median(I,2),a_dg(I,2),10);

