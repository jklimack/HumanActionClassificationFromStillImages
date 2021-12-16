% Human Action classification algorithm. 
%
% Three main stages to the algorithm: preprocessing images, feature 
%   extraction, and classification. 

clear all

% paths to data
path_train = "Dataset/TrainSet";
path_test = "Dataset/TestSet";

% retrieve the class names. The name of the directories in the train path
% are assigned as the class names. Therefore, all of the classes must have
% one directory in the train path, as well as the test path folders, and no
% other folders than these should exist. 
classes = getDirContents(path_train);

% select the feature extraction algorithm
feature_type = 'HOG'; % 'LBP'; % 

t0=clock();

% retrieve the train and test image file names, and their associated class
% labels. 
[train_files, trainy] = getImageFiles(path_train, classes);
[test_files, testy] = getImageFiles(path_test, classes);

% read, rescale, and convert to grayscale, all of the images in the train
% and test sets. 
train_imgs = readImages(train_files);
test_imgs = readImages(test_files);

% preprocess the images by applying histogram equalization and/or image
% smoothing. 
train_imgs = preprocessImages(train_imgs);
test_imgs = preprocessImages(test_imgs);

t1=clock();

% retrieve the feature vector for each image using one of the two
% implemented feature extraction algorithms. 
if feature_type=='HOG'
    trainx = getFeaturesHOG(train_imgs);
    testx = getFeaturesHOG(test_imgs);
else
    trainx = getFeaturesLBP(train_imgs);
    testx = getFeaturesLBP(test_imgs);
end

t2 = clock();

% train the model using an SVM
model = fitcecoc(trainx, trainy);

t3 = clock();

% predict the test set categories
predicty = predict(model, testx);

% output statistics
cmat = confusionmat(testy, predicty);
accuracy = sum(diag(cmat)) / sum(cmat(:))
confusionchart(cmat)

% Time outputs
t_image_read = t1-t0
t_feature_extraction = t2-t1
t_train = t3-t2


% ------------------------- END MAIN

 
function [image_files, labels]=getImageFiles(path, classes)
    image_files = {};
    idx = 1;
    for i = 1:length(classes)
       path_class = strcat(path, "/", classes{i});
       class_files = getDirContents(path_class);
       for j = 1:length(class_files)
           image_files{idx} = strcat(path_class, "/", class_files{j});
           labels{idx} = i;
           idx = idx+1;
       end
    end
    labels = cell2mat(labels);
end

function imgs=readImages(image_files)
    % read each image and store into a cell array, imgs. 
    
    % rescale images to a common size
    img_size = 256;
    
    imgs={};
    for i =1:length(image_files)
        img = imread(image_files{i});
        img = rgb2gray(img);
        img = imresize(img, [img_size, img_size]);
        imgs{i} = img;
    end
end

function content=getDirContents(path)
    % return a cell array listing the contents of a given path. 
    items = dir (path);
    items = items(3:length(items));
    content = {};
    for i = 1:length(items)
        content{i} = items(i).('name');
    end
end

function x=getFeaturesLBP(imgs)
    num_images = length(imgs);
    x = 0;
    for i=1:num_images
        img = imgs{i};
        featureVector = extractLBPFeatures(img, 'NumNeighbors', 8, 'Radius',1,'CellSize', [12,12]);
        if x==0
            x = zeros(num_images, length(featureVector));
        end
        x(i,:) = featureVector;
    end
end

function x=getFeaturesHOG(imgs)
    cell_size = [12,12];
    num_images = length(imgs);
    x=0;
    for i=1:num_images
        img = imgs{i};
        [featureVector,hogVisualization] = extractHOGFeatures(img, 'CellSize', cell_size, 'BlockSize', [2,2], 'NumBins', 9);
        if x ==0 
            x = zeros(num_images, length(featureVector));
        end
        x(i,:) = featureVector;
    end
end
 
function imgs=preprocessImages(imgs_init)
    for i=1:length(imgs_init)
        % Histogram Equalization (HE)
        %imgs_init{i} = histeq(imgs_init{i});
        % Smoothing
        %imgs_init{i} = imgaussfilt(imgs_init{i}, 2);
        %imgs_init{i} = medfilt2(imgs_init{i});
    end
    imgs = imgs_init;
end





