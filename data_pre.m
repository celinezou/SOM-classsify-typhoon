% clear;
% load coastlines
% coastlon(coastlon<0)=coastlon(coastlon<0)+360;
% h=plot(coastlon,coastlat,'-');
% axis([0 360 -90 90]);
clear all
%load ./data/color30
%load ./data/rainbow
figurewithtopo=false;
pacificorglobal=false;

%% topograpy
%samplefactor = 40;
%[Z, refvec] = etopo('./data/etopo1_ice_c_f4.flt', samplefactor);


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
weird_ind=[4553 5514 6239 6329 6331];
% classfied TS by wind >34kn TC_numObs>0
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
TCdate_ind=find(TCdate_all(1,:,1)>=1980&TCdate_all(1,:,1)<=2015);

% anytime in Pacific
if(pacificorglobal)
    basin(isnan(TC_wind))=nan;
    [row,col]=find(basin==2|basin==3|basin==11);
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

%% interp1
% %% figure with or without topo
figure; hold on
% mark=['+','*','.','x','d','s'];
% color=rainbow(1:(length(rainbow)-1)/(max(yc)-1):length(rainbow),:)/256;
 if (figurewithtopo)
    % figure with topo
    worldmap world
    setm(gca,'MapProjection','robinson')
    setm(gca,'Origin',[0,180,0]);
    setm(gca,'ParallelLabel','off');
    setm(gca,'MLabelLocation',60);
    setm(gca,'MLabelParallel','south');
    setm(gca,'MLabelParallel','south');
    setm(gca,'MapLatLimit',latlim,'MapLonLimit',lonlim)
    geoshow(Z, refvec, 'DisplayType', 'texturemap');
    demcmap(Z, 256);
    geoshow('landareas.shp', 'FaceColor', 'none', ...
       'EdgeColor', 'black');    

points=10.0;          % interp to 10 points
lat_interp=zeros(points,datanum);
lon_interp=zeros(points,datanum);
wind_interp=zeros(points,datanum);
TC_year=zeros(datanum,1);
method_int='linear';
for i=1:length(ind)
    time_index=TC_first(ind(i)):TC_first(ind(i))+TC_numObs(ind(i))-1;
    interp_ind=TC_first(ind(i)):(TC_numObs(ind(i))-1)/(points-1):TC_first(ind(i))+TC_numObs(ind(i))-1;
    lat_interp(:,i)=interp1(time_index,lat_wmo(time_index,ind(i)),interp_ind,method_int);
    lon_interp(:,i)=interp1(time_index,lon_wmo(time_index,ind(i)),interp_ind,method_int);
    wind_interp(:,i)=interp1(time_index,wind_wmo(time_index,ind(i)),interp_ind,method_int);
    TC_year(i)=TCdate_all(1,ind(i),1);
    plotm(lat_wmo(time_index,ind(i)),lon_wmo(time_index,ind(i)),'g');%interp
end

 end
% %% figure with or without topo
figure; hold on
% mark=['+','*','.','x','d','s'];
% color=rainbow(1:(length(rainbow)-1)/(max(yc)-1):length(rainbow),:)/256;
if (figurewithtopo)
    % figure with topo
    worldmap world
    setm(gca,'MapProjection','robinson')
    setm(gca,'Origin',[0,180,0]);
    setm(gca,'ParallelLabel','off');
    setm(gca,'MLabelLocation',60);
    setm(gca,'MLabelParallel','south');
    setm(gca,'MLabelParallel','south');
    setm(gca,'MapLatLimit',latlim,'MapLonLimit',lonlim)
    geoshow(Z, refvec, 'DisplayType', 'texturemap');
    demcmap(Z, 256);
    geoshow('landareas.shp', 'FaceColor', 'none', ...
       'EdgeColor', 'black');    

   
%     for j =1:length(ind)
            plotm(lat_wmo(:,ind),lon_wmo(:,ind),'g');%interp
%     end


figure; hold on
% mark=['+','*','.','x','d','s'];
% color=rainbow(1:(length(rainbow)-1)/(max(yc)-1):length(rainbow),:)/256;
% if (figurewithtopo)
    % figure with topo
    worldmap world
    setm(gca,'MapProjection','robinson')
    setm(gca,'Origin',[0,180,0]);
    setm(gca,'ParallelLabel','off');
    setm(gca,'MLabelLocation',60);
    setm(gca,'MLabelParallel','south');
    setm(gca,'MLabelParallel','south');
    setm(gca,'MapLatLimit',latlim,'MapLonLimit',lonlim)
    geoshow(Z, refvec, 'DisplayType', 'texturemap');
    demcmap(Z, 256);
    geoshow('landareas.shp', 'FaceColor', 'none', ...
       'EdgeColor', 'black');    

   
%     for j =1:length(ind)
            plotm(lat_interp(:,:),lon_interp(:,:),'g');%interp
%     end