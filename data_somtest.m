% clear;
% load coastlines
% coastlon(coastlon<0)=coastlon(coastlon<0)+360;
% h=plot(coastlon,coastlat,'-');
% axis([0 360 -90 90]);
clear
load color30
figurewithtopo=false;

%% topograpy
samplefactor = 10;
[Z, refvec] = etopo('etopo1_ice_c_f4.flt', samplefactor);


%% load data & plot original trace
FILE='Allstorms.ibtracs_wmo.v03r09.nc';
lat_wmo=ncread(FILE,'lat_wmo');
lon_wmo=ncread(FILE,'lon_wmo');
pres_wmo=ncread(FILE,'pres_wmo');
time_wmo=ncread(FILE,'time_wmo');
numObs=ncread(FILE,'numObs');
wind_wmo=ncread(FILE,'wind_wmo');
nature_wmo=ncread(FILE,'nature_wmo');
pres_wmo(pres_wmo<=0)=nan;
lon_wmo(lon_wmo<0)=lon_wmo(lon_wmo<0)+360;

%% choosing data
% weird tc
% weird_ind=[4553 5514 6239 6331];
weird_ind=[4553 5514 6239 6329 6331];
% classfied TS by wind 
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

% convert date to vector format
TCdate_all=reshape(datevec(reshape(time_wmo,[],1)+datenum('1858-11-17')),137,7267,6);
TCdate_ind=find(TCdate_all(1,:,1)>=1980&TCdate_all(1,:,1)<=2010);

% chose perdefined data
ind=intersect(find(TC_first>0),find(TC_numObs>9));
ind=intersect(ind,TCdate_ind);
ind=setdiff(ind,weird_ind);
datanum=length(ind);

%% interp1
points=10.0;          % interp to 10 points
lat_interp=zeros(points,datanum);
lon_interp=zeros(points,datanum);
for i=length(ind)-datanum+1:length(ind)
    time_index=TC_first(ind(i)):TC_first(ind(i))+TC_numObs(ind(i))-1;
    interp_ind=TC_first(ind(i)):(TC_numObs(ind(i))-1)/(points-1):TC_first(ind(i))+TC_numObs(ind(i))-1;
    lat_interp(:,i)=interp1(time_index,lat_wmo(time_index,ind(i)),interp_ind);
    lon_interp(:,i)=interp1(time_index,lon_wmo(time_index,ind(i)),interp_ind);
    wind_interp(:,i)=interp1(time_index,TC_wind(time_index,ind(i)),interp_ind);
end

lat_interp(lat_interp>0)=lat_interp(lat_interp>0)+180;


%% SOM
rng('default');
P=cat(1,lat_interp,lon_interp,wind_interp);
net=selforgmap([4 5],100,3);
net.trainparam.epochs=1000;
net=train(net,P);
y=net(P);
yc=vec2ind(y);
figure;plotsomhits(net,P);

lat_interp(lat_interp>0)=lat_interp(lat_interp>0)-180;

%% som results 
for j =1:max(yc)
    clas{j}=find(yc==j);
end

for j=1:max(yc)
    ave_lat(:,j)=mean(lat_interp(:,clas{j}),2);
    ave_lon(:,j)=mean(lon_interp(:,clas{j}),2);
end

legendInfo=cell(max(yc),1);
%% figure with or without topo
figure; hold on
if (figurewithtopo)
    % figure with topo
     worldmap world
    setm(gca,'Origin',[0,180,0]);
    setm(gca,'MLabelLocation',90);
    setm(gca,'MLabelParallel','south');
    geoshow(Z, refvec, 'DisplayType', 'texturemap');
    demcmap(Z, 256);
    geoshow('landareas.shp', 'FaceColor', 'none', ...
       'EdgeColor', 'black');    
    mark=['+','*','.','x','d','s'];
    color=color30;
    
    for j =1:max(yc)
        if (~isempty(clas{j}))
            plotm(lat_interp(:,clas{j}),lon_interp(:,clas{j}),'Color',color(j,:));%interp
        end
    end
    % figure; hold on;
    tmp2=[];
    for j=1:max(yc)
        if (~isempty(clas{j}))
            tmp2=[tmp2 j];
            %       %h(j)=plot(ave_lon(:,j),ave_lat(:,j),'LineWidth',2,'Color',color(j,:),'Marker','d','MarkerEdgeColor',color(j,:),'MarkerFaceColor',[0,0,0]);
            h(j)=plotm(ave_lat(:,j),ave_lon(:,j),'k-d','LineWidth',1.5,'MarkerEdgeColor',[0,0,0],'MarkerFaceColor',color(j,:));
            legendInfo{j} = ['C' num2str(j) ' num=' num2str(length(clas{j}))];
        end
    end

    legend(h(tmp2),legendInfo{tmp2},'Location','northwest')

else
    
    % figure without topo
    set(gca,'dataaspectratio',[1 1 1]);
    axis([0 360 -90 90])
    mark=['+','*','.','x','d','s'];
    color=color30;
    
    for j =1:max(yc)
        if (~isempty(clas{j}))
            plot(lon_interp(:,clas{j}),lat_interp(:,clas{j}),'Color',color(j,:));%interp
        end
    end
    % figure; hold on;
    tmp2=[];
    for j=1:max(yc)
        if (~isempty(clas{j}))
            tmp2=[tmp2 j];
            %       %h(j)=plot(ave_lon(:,j),ave_lat(:,j),'LineWidth',2,'Color',color(j,:),'Marker','d','MarkerEdgeColor',color(j,:),'MarkerFaceColor',[0,0,0]);
            h(j)=plot(ave_lon(:,j),ave_lat(:,j),'k-d','LineWidth',1.5,'MarkerEdgeColor',[0,0,0],'MarkerFaceColor',color(j,:));
            legendInfo{j} = ['C' num2str(j) ' num=' num2str(length(clas{j}))];
        end
    end
    
    legend(h(tmp2),legendInfo{tmp2},'Location','northwest')
end

