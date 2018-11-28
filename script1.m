clear all;
load 16color
pacificorglobal=true ;
figurewithtopo=true;
normalization=true;
%% load data & plot original trace
FILE='Allstorms.ibtracs_wmo.v03r09.nc';
lat_wmo=ncread(FILE,'lat_wmo');
lon_wmo=ncread(FILE,'lon_wmo');
pres_wmo=ncread(FILE,'pres_wmo');
time_wmo=ncread(FILE,'time_wmo');
numObs=ncread(FILE,'numObs');
wind_wmo=ncread(FILE,'wind_wmo');
nature_wmo=ncread(FILE,'nature_wmo');
basin=ncread(FILE,'basin');
pres_wmo(pres_wmo<=0)=nan;
lon_wmo(lon_wmo<0)=lon_wmo(lon_wmo<0)+360;


%% choosing data
% weird tc
% weird_ind=[4553 5514 6239 6331];
weird_ind=[4553 5514 6239 6329 6331,6863];
% classfied TS by wind >34kn TC_numObs>9
TC_wind=reshape(wind_wmo,1,[]);
TC_wind(TC_wind<34)=nan;
TC_wind=reshape(TC_wind,137,[]);
TC_first=ones(length(numObs),1);
for i=1:length(numObs)
    tmp1=find(isfinite(TC_wind(:,i)),1);
    if(isempty(tmp1))
        TC_first(i)=nan;
    else
        TC_first(i)=tmp1;
    end        
end
TC_numObs=nansum(TC_wind./TC_wind,1);
TCwind_ind=intersect(find(TC_first>0),find(TC_numObs>9));

% convert date to vector format 1980-2015
TCdate_all=reshape(datevec(reshape(time_wmo,[],1)+datenum('1858-11-17')),137,7267,6);
TCdate_ind=find(TCdate_all(1,:,1)>=1998&TCdate_all(1,:,1)<=2014);

% anytime in Pacific
if(pacificorglobal)
    basin(isnan(TC_wind))=nan;
    [row,col]=find(basin==2);
    TCbasin_ind=unique(col);
    latlim = [0 60];
    lonlim = [100 270];
else
    latlim = [-70 70];
    lonlim = [0 360];
    TCbasin_ind=1:length(numObs);
end

% chose perdefined data
ind=intersect(TCwind_ind,TCdate_ind);
ind=intersect(ind,TCbasin_ind);
ind=setdiff(ind,weird_ind);
% missing wind data 
TCwind_mis=[];
for i=1:length(ind)
    if sum(isnan(TC_wind(TC_first(ind(i)):TC_first(ind(i))+TC_numObs(ind(i))-1,ind(i))))>0
        TCwind_mis=[TCwind_mis;i];
    end
end
ind=setdiff(ind,ind(TCwind_mis));

datanum=length(ind);
%% interp spline
points=10.0;          % interp to 10 points
lat_interp=zeros(points,datanum);
lon_interp=zeros(points,datanum);
wind_interp=zeros(points,datanum);
TC_year=zeros(datanum,1);
method_int='spline';
for i=1:length(ind)
    time_index=TC_first(ind(i)):TC_first(ind(i))+TC_numObs(ind(i))-1;
    interp_ind=TC_first(ind(i)):(TC_numObs(ind(i))-1)/(points-1):TC_first(ind(i))+TC_numObs(ind(i))-1;
    lat_interp(:,i)=interp1(time_index,lat_wmo(time_index,ind(i)),interp_ind,method_int);
    lon_interp(:,i)=interp1(time_index,lon_wmo(time_index,ind(i)),interp_ind,method_int);
    wind_interp(:,i)=interp1(time_index,wind_wmo(time_index,ind(i)),interp_ind,method_int);
    TC_year(i)=TCdate_all(1,ind(i),1);
end
if(normalization)
    lat_interp0=lat_interp-repmat(lat_interp(1,:),points,1);
    lon_interp0=lon_interp-repmat(lon_interp(1,:),points,1);
else
    lat_interp0=lat_interp;
    lon_interp0=lon_interp;
end

%% SOM
rng('default');
P=cat(1,lat_interp0,lon_interp0,wind_interp);
net=selforgmap([3,3]);
net.trainParam.epochs=1000;
net=train(net,P);
y=net(P);
yc=vec2ind(y);
figure;plotsomhits(net,P);
%% som results 
for j =1:max(yc)
    clas{j}=find(yc==j);
end
%% plot
p1=figure();
   legendInfo=cell(max(yc),1);
% figure;
% set(gca,'dataaspectratio',[1 1 1]);
% axis([0 360 -90 90]);

tmp2=[];
for j=1:max(yc)
    if (~isempty(clas{j}))
        tmp2=[tmp2 j];
        ave_lat(:,j)=mean(lat_interp0(:,clas{j}),2);
        ave_lon(:,j)=mean(lon_interp0(:,clas{j}),2);
        h(j)=plot(ave_lon(:,j),ave_lat(:,j),'color',cmap(j,:),'linewidth',1);
        legendInfo{j} = ['C' num2str(j) ' num=' num2str(length(clas{j}))];
        hold on;
    end
end
legend(h(tmp2),legendInfo{tmp2},'Location','northwest');
new_fig_handle = shift_axis_to_origin( gca );
%=findobj(p1,'tag','legend');
if (figurewithtopo)
    % figure with topo
    figure();
    samplefactor = 10;
    [Z, refvec] = etopo('etopo1_ice_c_f4.flt', samplefactor);
    worldmap world
    setm(gca,'MapProjection','robinson');
    setm(gca,'Origin',[0,180,0]);
    setm(gca,'ParallelLabel','off');
    setm(gca,'MLabelLocation',60);
    setm(gca,'MLabelParallel','south');
    setm(gca,'MLabelParallel','south');
    setm(gca,'MapLatLimit',latlim,'MapLonLimit',lonlim);
    geoshow(Z, refvec, 'DisplayType', 'texturemap');
    demcmap(Z, 256);
    geoshow('landareas.shp', 'FaceColor', 'none', ...
       'EdgeColor', 'black');    

    for j=1:max(yc)
            if (~isempty(clas{j}))
            pp{j}=plotm(lat_interp(:,clas{j}),lon_interp(:,clas{j}),'color',cmap(j,:));
            pp2(j)=pp{j}(1,1);
            hold on;
        end
    end
 
legend(pp2(tmp2),legendInfo{tmp2},'Location','northwest');
end


