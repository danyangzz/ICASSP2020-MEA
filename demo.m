clc
close all

%%
addpath('data'); addpath('functions');
Files = dir(fullfile('data', '*.mat'));
Max_datanum = length(Files);

%% 
for data_num = 3:Max_datanum   
    Dname = Files(data_num).name;
    disp(['***********The test data name is: ***' num2str(data_num) '***'  Dname '****************'])
    load(Dname);
     
    file_path = 'Results/';
    folder_name = Dname(1:end-4);  
    file_path_name = strcat(file_path,folder_name);
    if exist(file_path_name,'dir') == 0   
       mkdir(file_path_name);
    end
    file_mat_path = [file_path_name '/'];
    
    k = 10; lambda = 0:0.01:10;
    time_MEA_PKN = zeros(length(k),length(lambda));
    time_MEA_Gauss = zeros(length(k),length(lambda));
    MEA_PKN_result = zeros(7,length(k),length(lambda));
    MEA_Gauss_result = zeros(7,length(k),length(lambda));
    for k_i = 1:length(k)
        knn = k(k_i);
        for lambda_i = 1:length(lambda)
            lambda_value = lambda(lambda_i);
   
            tic
            [~,y_PKN] = MEA(X,Y,knn,0,lambda_value);
            time_MEA_PKN(k_i,lambda_i) = toc;
            result_MEA_PKN = ClusteringMeasure(Y,y_PKN);
            MEA_PKN_result(:,k_i,lambda_i) = result_MEA_PKN'; 
            
            tic
            [~,y_Gauss] = MEA(X,Y,knn,1,lambda_value);
            time_MEA_Gauss(k_i,lambda_i) = toc;
            result_MEA_Gauss = ClusteringMeasure(Y,y_Gauss);
            MEA_Gauss_result(:,k_i,lambda_i) = result_MEA_Gauss'; 
            
            file_name = Dname;
            save ([file_mat_path,file_name],'Dname','MEA_PKN_result','MEA_Gauss_result','time_MEA_PKN','time_MEA_Gauss');
    
        end
    end  
end