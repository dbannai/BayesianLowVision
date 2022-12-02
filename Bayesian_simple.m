%% How do point estimate functions change with bias? 
clear; close all

%Prior mean and standard deviation
nu = 2;  
gamma = linspace(0.5,2,10);    

%Likelihood st.dev (generally factor lower than gamma)
sigma = 1;
F(numel(gamma)) = struct('cdata',[],'colormap',[]);

%Stimulus range and speed
ss = linspace(0,10,1000);       
stim = 7;

n_trials = 1000;
filename = "DiffGamma.gif";
for i = 1:numel(gamma)
    %Calculate analytic and numeric solutions for varying sigmas
    mapA = post_analytic(ss,stim,nu,gamma(i),sigma);
    mapsN = post_numeric(n_trials,ss,stim,nu,gamma(i),sigma);

    f = figure('visible','off'); hold on;
    xline(stim,"LineWidth",2,"Color","k",'LineStyle','--','DisplayName','Stimulus')
    plot(ss,normpdf(ss,nu,gamma(i)),'LineWidth',2,'Color','#eaa143','DisplayName','Prior')
    xline(nu,"LineWidth",2,"Color","#eaa143",'LineStyle','--','DisplayName','Prior Mean')
    
    plot(ss,mapA,'LineWidth',2,'Color','#3498db','DisplayName','Point Estimate')
    histogram(mapsN,20,'EdgeAlpha',0.3,'FaceAlpha',0.3,'FaceColor','#3498db','DisplayName','Numeric Soln',norm='pdf')
    text(0.5,1.25,{'Gamma:',sprintf('%.2f',gamma(i))},'FontSize',12)
    legend; ylim([0,1.5])
   
    %animate(filename,F,i)
end 

%% Psychometric function for Gaussian prior
clear; close all

nu = 5; gamma = 2;          %Prior mean, st.dev
sig1 = 0;                   %Measure 1 st.dev
sig2 = 1;                   %Measure 2 st.dev

ss = linspace(0,10,1000);
stim1 = 4; stim2 = 7;

%Calculate joint likelihood distribution and draw (Equation 30)
jlike = normpdf(ss,stim2,sig2)'*normpdf(ss,stim1,sig1);
%colormap(gray)
%imagesc(ss,ss,jlike); axis image; axis xy;
%xlabel("Measurement 1"); ylabel("Measurement 2")

like1 = normpdf(ss,stim1,sig1);
like2 = normpdf(ss,stim2,sig2);

%{
figure; hold on
plot(ss,normpdf(ss,nu,gamma),"LineWidth",2,"Color","k",'LineStyle','--','DisplayName','Prior')
plot(ss,like1,"LineWidth",2,"Color","#3498db",'DisplayName','Likelihood 1')
plot(ss,like2,"LineWidth",2,"Color","#eaa143",'DisplayName','Likelihood 2')
legend
xlabel("Stimulus Speed"); ylabel("Probability")
%}

%Create psychometric function across values of x2 (Equation 33)
sig1 = 3;
%sig1 = [0.5:0.5:4];
F(numel(sig1)) = struct('cdata',[],'colormap',[]);
stim2 = linspace(0,10,1000);
psychF = zeros(size(stim2));

for j = 1:numel(sig1)
    for i = 1:numel(stim2)
        alpha1 = alpha_calc(gamma,sig1(j));
        alpha2 = alpha_calc(gamma,sig2);

        psychF(i) = phi(alpha1,alpha2,stim1,stim2(i),sig1,sig2(j)); 
    end
    
    f = figure('visible','on');
    plot(ss,psychF,"LineWidth",2,"Color","k")
    xline(stim1,"LineWidth",2,"Color","#eaa143",'LineStyle','--','DisplayName','Stimulus 1')
    xlabel("Stimulus 2, (Stimulus 1 = 4)"); ylabel("Probability of discernment")
    legend
    %animate("Diff2.gif",F,j)
    
end 


%% Define your functions
function alpha = alpha_calc(gamma,sigma)             %Shrinkage of posterior
   alpha = (gamma^2)/(gamma^2 + sigma^2);
end 

function nT = nu_tilda(nu,gamma,sigma)      %Offset of posterior
    nT = nu*((sigma^2)/(gamma^2 + sigma^2));
end 

function mapA = post_analytic(ss,stim,nu,gamma,sigma)
    alpha = alpha_calc(gamma,sigma);
    nT = nu_tilda(nu,gamma,sigma);
    mapA = normpdf(ss, alpha*stim + nT, sqrt(alpha^2*sigma^2));
end 

function mapsN = post_numeric(n_trials,ss,stim,nu,gamma,sigma)
    mapsN = zeros(n_trials,1);

    for t = 1:n_trials
        meas = stim + randn*sigma;
        post = normpdf(ss,nu,gamma).*normpdf(ss,meas,sigma);

        mapN = ss(post == max(post));
        if numel(mapN)>1
            mapN = mean(mapN);
        end 
    
        mapsN(t) = mapN;
    end 
end 

function animate(filename,F,i)
    drawnow
    F(i) = getframe();
    im = frame2im(F(i));
    [imind,cm] = rgb2ind(im,256);
    if i == 1
        imwrite(imind,cm,filename,'gif', 'Loopcount',inf);
    else
        imwrite(imind,cm,filename,'gif','WriteMode','append');
    end

end 

function F = phi(alpha1,alpha2,stim1,stim2,sig1,sig2)
    denom = sqrt((alpha2^2 * sig2^2)+(alpha1^2 * sig1^2));
    num = (alpha2 * stim2) - (alpha1*stim1);
    F = normcdf(num/denom);
end 
      