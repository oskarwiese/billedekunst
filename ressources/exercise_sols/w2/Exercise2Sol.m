%% Exercise 2 - Cameras
%% Ex2.1
clc;clear all;
% Theta can be calculated by theta= arctan(b/a)
a = 10; b = 3;
theta = atand(b/a);  %theta=16.70 degress (atand = arctan in degrees)

%% Ex2.2
% The function can be written as:

% function b = CameraBDistance(f,g)
% %CameraBDistance returns the distance (b) where the CCD should be placed
% %when the object distance (g) and the focal length (f) are given
% b = 1/(1/f -1/g);

% Use your function to find out where the CCD should be placed when the focal
% length is 15 mm and the object distance is 0.1, 1, 5, and 15 meters.

b1 = CameraBDistance(15,100);
disp(['With f=15mm and g=100mm we get b=',num2str(b1),'mm'])
b2 = CameraBDistance(15,1000);
disp(['With f=15mm and g=1000mm we get b=',num2str(b2),'mm'])
b3 = CameraBDistance(15,5000);
disp(['With f=15mm and g=5000mm we get b=',num2str(b3),'mm'])
b4 = CameraBDistance(15,15000);
disp(['With f=15mm and g=15000mm we get b=',num2str(b4),'mm'])

% What happens to the place of the CCD when the object distance is increased?
% -> the place of the CCD converges to the size of the focal length
%% Ex2.3
%info = imfinfo('DTUSigns.jpg')
%info.DigitalCamera

%% Ex2.4
clc;clear all;
% we wish to use mm for our constants in this exercise

G = 1800; % Thomas' height in mm
f = 5; % Cameras focal length in mm
g = 5000; % Thomas' distance to camera in mm
pixels = 640*480; % number of pixels in camera
area = 4.8*6.4; % area of camera in mm^2

%% 1) A focused image of Thomas is formed inside the camera. At which 
% distance from the lens?
% -> We use our CameraBDistance function:
b = CameraBDistance(f,g);
sprintf('Thomas is formed at the distance %f mm inside the camera ',b)

%% 2)How tall (in mm) will Thomas be on the CCD-chip? 
B = RealSizeOnCCD(G,b,g);

% function B = RealSizeOnCCD(G,b,g)
% %Input:
% %  G = Real height of the object in mm
% %  b = the distance (b) where the CCD should be placed in mm
% %  g = the object distance in mm
% %Output:
% %  B = Size of obejct in lens in mm

% B = G*b/g;
sprintf('Thomas will have the hegiht %f mm on the CCD chip ',B)

%% 3) What is the size of a single pixel on the CCD chip? (in mm)?
% -> The size of a single pixel can be found by taking the whole chip area
% divided with the number of pixels in the chip:
pixelsize = area/pixels;
sprintf('The size of a single pixel is %f mm2 on the CCD chip ',pixelsize)

%% 4) How tall (in pixels) will Thomas be on the CCD-chip? 
% Since a pixel is square we can find the side-length of the pixel by
pixelheight = sqrt(pixelsize);

% Now can Thomas' height in pixels be found
Hp = PixelSizeOnCCD(G,b,g,pixelheight);

% function  Hp = PixelSizeOnCCD(G,b,g,pixelheight)
% % Input:
% %   G = Real height of the object in mm
% %   b = the distance (b) where the CCD should be placed in mm
% %   g = the object distance in mm
% %   pixelheight = height of a pixel in mm
% % Output:
% %   Hp = Height in pixels
% 
% B = RealSizeOnCCD(G,b,g);
% Hp = B/pixelheight;

sprintf('Thomas will be %f pixels tall ',Hp)

%% 5)What is the horizontal field-of-view (in degrees)?
% See illustration on page 17. To compute v we divide the width of the chip
% with 2
W = 6.4/2;
% Same concept as in Ex2.1
v1 = atand(W/f)*2;

sprintf('The horizontal field of view will have the angle of %f degrees',v1)

%% 6) What is the vertical field-of-view (in degrees)?
% We have to take half of the CCD height
H = 4.8/2;
v2 = atand(H/f)*2;

sprintf('The vertical field of view will have the angle of %f degrees',v2)

% function [horizontal, vertical] = CameraFOV(f,CCD_height,CCD_width)
% %Input
% %  f: focal length
% %  CCD_height: Height of the CCD chip
% %  CCD_width: Width of the CCD chip
% %Output:
% % horizontal: horizontal field-of-view
% % vertical: vertical field-of-view
%
% horizontal = atand((CCD_width/2) /f) *2;
% vertical = atand((CCD_height/2) /f) *2;

%% Exam question on camera geometry
clc; clear all;

f = 65; % in mm
imagesize = [5120 , 4096]; % Pixelsize
g = 1200; % Distance to camera in mm
CCDsize = [10,8]; % in mm
r = 400; % Radius of melanoma in pixels

% First we want to find the radius of the melanoma in mm
Pixeltomm = CCDsize(1)/imagesize(1);
B = Pixeltomm * r;
% Now we can use b/B = g/G to calculate G:
G = g/(f/B); 
% Now we have the radius of the physical object in mm. We can then find the
% area by:
Area = G^2*pi;

sprintf('The physical area of the melanoma is %f',Area) % Answer is option 1
%% Exam question of field-of-view
clc;clear all;
% We have to take half of v to get our triangle that calculates half of the 
% length of the finger. Then we multiply the result by 2 to get full length

v = 15/2; % Degrees
g = 31.5; % cm

length = tand(v)*g*2;

sprintf('The finger has the length of %.2f cm',length) % The answer is option 2





