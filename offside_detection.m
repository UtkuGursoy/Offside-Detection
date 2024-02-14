%% Initialization
clear
close all
clc

%% Read Video
obj = VideoReader('12.mp4'); % create video object
start_frame = 1;            % start frame
offside_frame = 3;          % frame at which the pass is made

for frame_index = 1:4       % parsing through video frames
    if(exist('img'))
        prev_img = img;
    end
    img = readFrame(obj);
    
    if(frame_index<start_frame)
        continue
    end
    
%% Getting Vanishing Point lines
    % Requesting parallel vanishing lines input from user
    if (frame_index == start_frame)
        imshow(img)
        title("Please select two points on field lines to calculate vanishing point");
        [x,y] = getpts;
        points = [x,y];
        close all
        %tStart = cputime;
    % Calculating Slope
    m = zeros(size(points,1)/2,1);          % slope matrix   
    c = zeros(size(points,1)/2,1);          % intercept matrix             
    k = 1;
    vp = zeros(2,1);                        % vanishing point matrix
    for j = 1:2:size(points,1)
        m(k) = (points(j+1,2) - points(j,2)) / (points(j+1,1) - points(j,1));
        c(k) = - points(j,1) * m(k) + points(j,2);
        k = k+1;
    end
    % Calculating the Vanishing point
    count = 0;
    for p = 1:size(points,1)/2
       for q = p+1:size(points,1)/2
           count = count + 1;
           A = [-m(p),1;-m(q),1]; 
           b = [c(p);c(q)];
           vp = vp + A\b;
       end
    end
    vp = int32(vp/count);
    end
    
%% Actual Detection starts (one every 20 frames).
    BW_img = rgb2gray(img);                 % Coverting the image to grayscale
    Edge_img = edge(BW_img,'sobel');        % Converting greyscale image to edge image using Sobel
    
% Removing the TOP Boundary using Hough Transform
%   Defining Hough Parameters
    start_angle = 89;                       
    end_angle = 89.99;
    theta_resolution = 0.01;
%   Obtaining Hough coefficients    
    [hou,theta,rho] = hough(Edge_img(1:floor(size(Edge_img,1)/2),:), 'Theta', start_angle:theta_resolution:end_angle);
    peaks = houghpeaks(hou,2,'threshold',ceil(0.3*max(hou(:))));
    lines = houghlines(Edge_img(1:floor(size(Edge_img,1)/2),:),theta,rho,peaks,'FillGap',5,'MinLength',7);
%   Identifying longest horizontal lines    
    min_row = lines(1).point1(2);
    xy_long = [lines(1).point1; lines(1).point2];
    
    max_len = 0;
    sizes = size(img);
    for k = 1:length(lines)
       xy = [lines(k).point1; lines(k).point2];
    
       % Determine the endpoints of the longest line segment
       len = abs(lines(k).point1(1) - lines(k).point2(1));
       if ( len > max_len)
          max_len = len;
          xy_long = xy;
       end
    end
    if(xy_long(2,1)-xy_long(1,1) > 70)
    %   Removing top boundary pixels
        img(1:xy_long(:,2)-10,:,:)=0;
        BW_img(1:xy_long(:,2)-10,:,:)=0;
        Edge_img(1:xy_long(:,2)-10,:,:)=0;
    end

%    
   
%% Determining the players and Team_Ids
    img_valid = img;
%   Define defending team colours
    indg = find(fuzzycolor(im2double(img_valid),'red')<0.1);
    n = size(img,1)*size(img,2);

    img_team_read = img_valid;
    img_team_read([indg;indg+n;indg+2*n]) = 0;
    
%   Image processing
    mask = imbinarize(rgb2gray(img_team_read));
    mask = imfill(mask,'holes'); 
    mask_open = bwareaopen(mask,30);
    mask_open = imfill(mask_open,'holes');
    if(frame_index == offside_frame)
        figure; sgtitle('Defending Team'); subplot(1,2,1);imshow(mask_open);title("Mask with 'imfill'");
    end
    S_E_D = strel('disk',15);
    mask_open = imdilate(mask_open,S_E_D);
    Conn_Comp_team_red = bwconncomp(mask_open,8);
    S_team_red = regionprops(Conn_Comp_team_red,'BoundingBox','Area');
    if(frame_index == offside_frame)
       subplot(1,2,2); imshow(mask_open);title("Mask with Disk Structuring Element");
    end

%   Define attacking team colours
    indg = find(fuzzycolor(im2double(img_valid),'blue')<0.1);
    n = size(img,1)*size(img,2);
    img_team_read = img_valid;
    img_team_read([indg;indg+n;indg+2*n]) = 0;
    mask = imbinarize(rgb2gray(img_team_read));
    mask = imfill(mask,'holes');
    mask_open = bwareaopen(mask,15);
    mask_open = imfill(mask_open,'holes'); 
    if(frame_index == offside_frame)
        figure; sgtitle('Attacking Team'); subplot(1,2,1);imshow(mask_open);title("Mask with 'imfill'");
    end
    S_E_D = strel('disk',15);
    mask_open = imdilate(mask_open,S_E_D); % dilate (genişletmek) the binary image with disk structuring element 
    Conn_Comp_team_blue = bwconncomp(mask_open,8);
    S_team_blue = regionprops(Conn_Comp_team_blue,'BoundingBox','Area');
    if(frame_index == offside_frame)
         subplot(1,2,2);imshow(mask_open);title("Mask with Disk Structuring Element");
    end

%   Getting all players/teamids in one list
    S = [S_team_red; S_team_blue];
    Team_Ids = [ones(size(S_team_red,1),1); 2*ones(size(S_team_blue,1),1)]; % Mark defense team as 1 and attacker team as 2
    Players = cat(2,[vertcat(S(1:size(S,1)).BoundingBox)], Team_Ids); % Concatenate team_ids with their corresponding teams
   
    %% Mark the bounding boxes
    f = figure('visible','off');    
    left_most = 9999;
    team_index = 5;

    if(frame_index == offside_frame)
        figure; imshow(img)
        hold on;
    end

    for i =1:size(S,1)                    % For all detected players
        BB = S(i).BoundingBox;            % Get their bounding box
        if(Team_Ids(i)==1)                % If defense team
            text(BB(1)-20, BB(2)-10,'D'); % Display D above its bounding box
            BB(4)  = 1.5*BB(4);
            S(i).BoundingBox(4) = BB(4);  % Bounding boxes of defense team is larger (?)
        end
        if(Team_Ids(i)==2)                % If attack team
            text(BB(1)-20, BB(2)-10,'A'); % Display A above its bounding box
        end
        rectangle('Position',[BB(1),BB(2),BB(3),BB(4)],...
        'LineWidth',2,'EdgeColor','red')  % Draw a rectangle to display bounding box

        x1 = floor(BB(1)+BB(3)/4);        % Coordinates of bounding box 
        y1 = floor(BB(2) + BB(4));
        ly = size(img,1);
        slope = int32((vp(2) - y1)/(vp(1) - x1));% Slope of the line from vanishing point to bounding box
        y_int = int32(- x1 * slope + y1);
        lx = int32((ly - y_int)/slope);          % Left x coordinate of the player
        if(lx<left_most && Team_Ids(i) == 1) % Determine x coord of the leftmost defender
            left_most = lx;
            slope_last_def = slope;
            y_int_last_def = y_int;
        end         
    end
    
    offside = 0;
    if(frame_index == offside_frame)
        m = (ly - vp(2))/(left_most-vp(1));
        c = vp(2) - m*vp(1);
        for z = 1:size(Players,1) 
            if(Players(z,team_index) == 2)
                x = Players(z,1) + Players(z,3)/4;
                y = Players(z,2) + Players(z,4);
                if(m*x + c > y)
                    title("Offside detected")
                    offside=1;
                    break
                end
            end
        end
        if offside == 0
            title("Failed to Detect Offside")
        end
    end

%   Plot offside Line
    plot([left_most,vp(1)],[ly ,vp(2)],'c','LineWidth',1)
 %{
    %% Mark the bounding boxes
    f = figure('visible','off');    
    left_most = 9999;
    team_index = 5;
    [~,last_player] = min(Players(:,1));
    
% Detecting if furthest attacking player is offside
    if(frame_index == offside_frame && Players(last_player, team_index)==2)
        % If the last player is attacker, it is offside
        disp("Offside detected")
        figure; imshow(img)
        hold on;
    elseif(frame_index == offside_frame && Players(last_player, team_index)==1)
        % If the last player is defense, it is not offside
        disp("Failed to Detect Offside")
        figure; imshow(img)
        hold on;
    end
    
    for i =1:size(S,1)                    % For all detected players
        BB = S(i).BoundingBox;            % Get their bounding box
        if(Team_Ids(i)==1)                % If defense team
            text(BB(1)-20, BB(2)-10,'D'); % Display D above its bounding box
            BB(4)  = 1.5*BB(4);
            S(i).BoundingBox(4) = BB(4);  % Bounding boxes of defense team is larger (?)
        end
        if(Team_Ids(i)==2)                % If attack team
            text(BB(1)-20, BB(2)-10,'A'); % Display A above its bounding box
        end
        rectangle('Position',[BB(1),BB(2),BB(3),BB(4)],...
        'LineWidth',2,'EdgeColor','red')  % Draw a rectangle to display bounding box

        x1 = floor(BB(1)+BB(3)/4);        % Coordinates of bounding box 
        y1 = floor(BB(2) + BB(4));
        ly = size(img,1);
        slope = int32((vp(2) - y1)/(vp(1) - x1));% Slope of the line from vanishing point to bounding box
        y_int = int32(- x1 * slope + y1);
        lx = int32((ly - y_int)/slope);          % Left x coordinate of the player
        if(lx<left_most && Team_Ids(i) == 1) % Determine x coord of the leftmost defender
            left_most = lx;
        end         
    end
%   Plot offside Line
    plot([left_most,vp(1)],[ly ,vp(2)],'c','LineWidth',1)
%}
end
%tEnd = cputime - tStart