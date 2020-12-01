%% to test some rotations
clear;close all;clc;
fileID=fopen('datafile.h','w','n','UTF-8')% open as write, native, utf8

x=randn(3,20);x=round(x*1000000000)/1000000000;

%fwrite(fileID, xsave, 'double')% binary, check native or not endian etc
fprintf(fileID,'#pragma once\n');
fprintf(fileID,'#include \"vectorsAndMatrixes.h\"\n');
fprintf(fileID,'//this file contains test data for the rotation unit tests\n');
fprintf(fileID,'namespace mlib{\n');
fprintf(fileID,'namespace testdata{\n');
fprintf(fileID,'namespace rotation{\n');
%std::vector<Vector3> vs={Vector3(1,1,1),Vector3(1,1,1),Vector3(1,1,1)};
%works
fprintf(fileID,'std::vector<Vector3> x={');
for i=1:length(x)
    if(i==length(x))
        fprintf(fileID,'Vector3(%12.12d, %12.12d, %12.12d) ',x(:,i));
    else
        fprintf(fileID,'Vector3(%12.12d, %12.12d, %12.12d), ',x(:,i));
    end
end
fprintf(fileID,'};\n');

% first generate a list of x to test,


% then some rotations, no need to save them since I can gen them locally
% however in order to the the answers should be known!

angles=randn(3,20)*2*pi;
angles=round(angles*1000000000)/1000000000;
fprintf(fileID,'std::vector<Vector3> angles={');
for i=1:length(angles)
    if(i==length(angles))
        fprintf(fileID,'Vector3(%12.12d, %12.12d, %12.12d) ',angles(:,i));
    else
        fprintf(fileID,'Vector3(%12.12d, %12.12d, %12.12d), ',angles(:,i));
    end
end
fprintf(fileID,'};\n');

R=[];
for i=1:length(angles)
    Rx=[1,                      0,                  0;
        0,                      cos(angles(1,i)),   -sin(angles(1,i));
        0,                      sin(angles(1,i)),   cos(angles(1,i))];
    
    Ry=[cos(angles(2,i)),       0,                  sin(angles(2,i));
        0,                      1,                  0;
        -sin(angles(2,i)),      0,                  cos(angles(2,i))];
    
    Rz=[cos(angles(3,i)),-sin(angles(3,i)),0;
        sin(angles(3,i)),cos(angles(3,i)),0;
        0,0,1];
    R=[R;Rx*Ry*Rz];
end

fprintf(fileID,'std::vector<std::vector<Vector3> > xr={ ');


for i=0:length(R)/3-1
    Ri=R((i*3 +1):(3 +i*3),1:3);
    xr=Ri*x;
    fprintf(fileID,'{');
    for j=1:length(xr)
        if(j==length(xr))
            fprintf(fileID,'Vector3(%12.12d, %12.12d, %12.12d) ',xr(:,j));
        else
            fprintf(fileID,'Vector3(%12.12d, %12.12d, %12.12d), ',xr(:,j));
        end
    end
    if(i==(length(angles)-1))
        fprintf(fileID,'} ');
    else
        fprintf(fileID,'},\n ');
    end
end
fprintf(fileID,'};\n ');

fprintf(fileID,'}// end namespace mlib\n');
fprintf(fileID,'}// end namespace testdata\n');
fprintf(fileID,'}//end namespace rotation\n');
fclose(fileID)

%%


clear;clc;close all;
th=pi/2;


Ry=[cos(th),       0,                  sin(th);
    0,                      1,                  0;
    -sin(th),      0,                  cos(th)];

Rx=[1,                      0,                  0;
    0,                      cos(th),   -sin(th);
    0,                      sin(th),   cos(th)];



Rz=[cos(th),-sin(th),0;
    sin(th),cos(th),0;
    0,0,1];

x=[1;2;3];
xr=Rx*x
yr=Ry*x
zr=Rz*x


%%
R=[-0.998834, 0.0482679, 0.00011296;
                   0.0482342, 0.998044, 0.0397666;
                   0.00180671, 0.0397257, -0.999209]
               det(R)
%%
    syms a b c d real;
    k=[ a, 0, b;
    0, c, d;
    0, 0, 1]

    ki=[ 1/a,   0, -b/a;
      0, 1/c, -d/c;
       0,   0,    1]
   
   
   
   
   
   
   
   
   
   
%%
x=
