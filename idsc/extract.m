addpath common_innerdist;
clear;

%------ Parameters ----------------------------------------------
ifig		= 1;
sIms		= importdata('cal_list.txt');
sImn        = importdata('cal_name.txt');
save_path   = '/sharefiles1/chenxy/cal101_final/saved_shape_context_9x16/';
% base='/sharefiles1/chenxy/cal101_final/saved_shape_context_5x12/';
% {'./mpeg_png/apple-1.png'}
% {'shape_context1'}

%-- shape context parameters
n_dist      = 9;
n_theta     = 16;
bTangent    = 1;
bSmoothCont	= 1;
n_contsamp	= 150;

%-- Extract inner-distance shape context
% figure(ifig);	clf; hold on;	set(ifig,'color','w'); colormap(gray);
for k=570000:length(sIms)
%     fn=strcat(base,sImn{k});
%     if exist(fn)==0
%         continue;
%     end
    file_name = strcat(save_path, sImn{k});
    flag=exist(file_name);
    if flag~=0
        continue;
    end
	
	%- Contour extraction
	ims	= double(imread(sIms{k}));
    ims = ims(:, :, 1);
	Cs	= extract_longest_cont(ims, n_contsamp);
	
	%- inner-dist shape context
	msk		= ims;%>.5;
	[sc,V,E,dis_mat,ang_mat] = compu_contour_innerdist_SC( ...
									Cs,msk, ...
									n_dist, n_theta, bTangent, bSmoothCont,...
									0);
%     file_name = strcat(save_path, sImn{k});
%     file_name = strcat(file_name, '.png')
    imwrite(sc, file_name);
	k
end

