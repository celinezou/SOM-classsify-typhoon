% clear;
% load coastlines
% coastlon(coastlon<0)=coastlon(coastlon<0)+360;
% h=plot(coastlon,coastlat,'-');
% axis([0 360 -90 90]);
clear
%% topograpy
samplefactor = 10;
[Z, refvec] = etopo('etopo1_ice_c_f4.flt', samplefactor);
% figure
% worldmap world
% setm(gca,'Origin',[0,180,0]);
% setm(gca,'MLabelLocation',90);
% setm(gca,'MLabelParallel','south');
% geoshow(Z, refvec, 'DisplayType', 'texturemap');
% demcmap(Z, 256);
% geoshow('landareas.shp', 'FaceColor', 'none', ...
%    'EdgeColor', 'black');

%% load data & plot original trace
FILE='Allstorms.ibtracs_wmo.v03r04.nc';
lat_wmo=ncread(FILE,'lat_wmo');
lon_wmo=ncread(FILE,'lon_wmo');
pres_wmo=ncread(FILE,'pres_wmo');
numObs=ncread(FILE,'numObs');
wind_wmo=ncread(FILE,'wind_wmo');
nature_wmo=ncread(FILE,'nature_wmo');
pres_wmo(pres_wmo<=0)=nan;
lon_wmo(lon_wmo<0)=lon_wmo(lon_wmo<0)+360;
%cm=[flipud(hot(256));0 0 0];
c=hot(256);
cm=[c(64:192,:);0 0 0];
pos=ceil((pres_wmo-min(nanmin(pres_wmo)))./(max(nanmax(pres_wmo))-min(nanmin(pres_wmo))).*129);
pos(find(pos==0))=1;
pos(isnan(pos))=130;
% for j=6600:6823
% scatterm(lat_wmo(:,j),lon_wmo(:,j),8,cm(pos(:,j),:),'filled');
% end

% choosing data 
rec10=find(numObs>=10);
[row,nature0]=find(nature_wmo==0|nature_wmo==5);
nature0=unique(nature0);
ind=intersect(rec10,nature0);

% interp1
for i=length(ind)-2000:length(ind)
    numi=double(numObs(ind(i)));
    % interp to 10 points
lat_interp(:,i)=interp1(1:numi,lat_wmo(1:numi,ind(i)),1:(numi-1)/9:numi);
lon_interp(:,i)=interp1(1:numi,lon_wmo(1:numi,ind(i)),1:(numi-1)/9:numi);

end

% plot interp trace
% for j=length(ind)-100:length(ind)
% plotm(lat_interp(:,j),lon_interp(:,j),'r');
% end

%% SOM
P=cat(1,lat_interp(:,end-2000:end),lon_interp(:,end-2000:end));
net=selforgmap([3 3]);
net.trainparam.epochs=1000;
net=train(net,P);
y=net(P);
yc=vec2ind(y);
figure;plotsomhits(net,P);
% plot

figure; worldmap world
setm(gca,'Origin',[0,180,0]);
setm(gca,'MLabelLocation',90);
setm(gca,'MLabelParallel','south');
geoshow(Z, refvec, 'DisplayType', 'texturemap');
demcmap(Z, 256);
geoshow('landareas.shp', 'FaceColor', 'none', ...
   'EdgeColor', 'black');
mark=['+','*','.','x','d','s'];
color=['y','m','c','k','r','w','g','r','k'];
for j =1:9
    clas{j}=find(yc==j);

     if ~isempty(clas{j})
%    h=plotm(lat_interp(:,end-1001+clas),lon_interp(:,end-1001+clas),mark(j),...
%             'MarkerFaceColor','r','MarkerEdgeColor','r','MarkerSize',4);hold on;
       h=plotm(lat_interp(:,end-2001+clas{j}),lon_interp(:,end-2001+clas{j}),color(j));hold on;%interp
%      h=plotm(lat_wmo(:,ind(end-2001+clas)),lon_wmo(:,ind(end-2001+clas)),color(j));hold on;
     end
end

% for j=1:4 clas{j}=find(yc==j);end;
% h1=plotm(lat_interp(:,end-2001+clas{1}),lon_interp(:,end-2001+clas{1}),color(1));hold on;
% h2=plotm(lat_interp(:,end-2001+clas{2}),lon_interp(:,end-2001+clas{2}),color(2));hold on;
% h3=plotm(lat_interp(:,end-2001+clas{3}),lon_interp(:,end-2001+clas{3}),color(3));hold on;
% h4=plotm(lat_interp(:,end-2001+clas{4}),lon_interp(:,end-2001+clas{4}),color(4));hold on;
% legend([h1(1,:) h2(1,:) h3(1,:) h4(1,:)],'1','2','3','4','Location','northeastoutside');
for j=1:9
ave_lat(:,j)=mean(lat_interp(:,end-2001+clas{j}),2);
ave_lon(:,j)=mean(lon_interp(:,end-2001+clas{j}),2);
end
plotm(ave_lat,ave_lon,'w-*');

