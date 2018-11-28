% clear;
% load coastlines
% coastlon(coastlon<0)=coastlon(coastlon<0)+360;
% h=plot(coastlon,coastlat,'-');
% axis([0 360 -90 90]);
clear
tcfile='Allstorms.ibtracs_wmo.v03r09.nc';


%% load data & plot original trace
lat_wmo=ncread(tcfile,'lat_wmo');
lon_wmo=ncread(tcfile,'lon_wmo');
lon_wmo(lon_wmo<0)=lon_wmo(lon_wmo<0)+360;
pres_wmo=ncread(tcfile,'pres_wmo');
numObs=ncread(tcfile,'numObs');
nature_wmo=ncread(tcfile,'nature_wmo');
time_wmo=ncread(tcfile,'time_wmo');
wind_wmo=ncread(tcfile,'wind_wmo');
pres_wmo(pres_wmo<=0)=nan;



%% choosing data 
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

ind=intersect(find(TC_first>0),find(TC_numObs>6));
ind=intersect(ind,TCdate_ind);
datanum=length(ind);

% interp1
points=10.0;          % interp to 10 points
lat_interp=zeros(points,datanum);
lon_interp=zeros(points,datanum);
for i=length(ind)-datanum+1:length(ind)
    time_index=TC_first(ind(i)):TC_first(ind(i))+TC_numObs(ind(i))-1;
    interp_ind=TC_first(ind(i)):(TC_numObs(ind(i))-1)/(points-1):TC_first(ind(i))+TC_numObs(ind(i))-1;
    lat_interp(:,i)=interp1(time_index,lat_wmo(time_index,ind(i)),interp_ind);
    lon_interp(:,i)=interp1(time_index,lon_wmo(time_index,ind(i)),interp_ind);
end
% nan in the middle
tmpr=[];
for i=1:length(ind)
 if sum(isnan(TC_wind(TC_first(ind(i)):TC_first(ind(i))+TC_numObs(ind(i))-1,ind(i))))>0;
     tmpr=[tmpr;i];
 end
end

% cross longitude 0 degree
ind3=[];
origin_lon=ncread(tcfile,'lon_wmo');
[r,l]=find(origin_lon>100|origin_lon<-100);
ind2=setdiff(ind,unique(l));
for i=1:length(ind2)
a=find(sign(origin_lon(:,ind2(i)))<0);
b=find(sign(origin_lon(:,ind2(i)))>0);
c=find(sign(origin_lon(:,ind2(i)))==0);
if (isempty(a)|isempty(b))& isempty(c)
ind3=[ind3;ind2(i)];
end
end
excludind=setdiff(ind2,ind3);
% plot interp trace
figure(2)
hold on;
set(gca,'dataaspectratio',[1 1 1]);
axis([0 360 -90 90])
for j=length(ind)-datanum+1:length(ind)
plot(lon_interp(:,j),lat_interp(:,j),'-r');
end

% % plot single trace
figure(3)
wcg_indi=tmpr;
plot(lon_wmo(:,wcg_indi),lat_wmo(:,wcg_indi))
set(gca,'dataaspectratio',[1 1 1]);
axis([0 360 -90 90])
% 
% % plot all trace
% figure(4)
% hold on;
% set(gca,'dataaspectratio',[1 1 1]);
% axis([-180 180 -90 90])
% for j=6823-datanum:6823
% plot(lon_wmo(:,j),lat_wmo(:,j),'.r');
% end

% 
% %% SOM
% P=cat(1,lat_interp(:,end-datanum:end),lon_interp(:,end-datanum:end));
% net=selforgmap([2 2]);
% net.trainparam.epochs=1000;
% net=train(net,P);
% y=net(P);
% yc=vec2ind(y);
% %figure(3);
% %plotsomhits(net,P);
% 
