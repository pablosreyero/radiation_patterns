clc
close all

msi_file = '80010465_0791_x_co.msi'
[Horizontal,Vertical] = msiread('80010465_0791_x_co.msi');
disp('Horizontal:')
disp([Horizontal.Azimuth, Horizontal.Magnitude])
disp('Vertical:')
disp([Vertical.Elevation, Vertical.Magnitude])

% vertical
figure()
P = polarpattern(Vertical.Elevation, Vertical.Magnitude, AngleAtTop=0, AngleDirection='cw');
P.TitleTop = 'Vertical';
createLabels(P,'Vertical');

% horizontal
figure()
P2 = polarpattern(Horizontal.Azimuth, Horizontal.Magnitude, AngleAtTop=0, AngleDirection='cw');
P2.TitleTop = 'Horizontal';
createLabels(P2,'Horizontal');

%3D pattern now
figure()
vertSlice = Vertical.Magnitude;
theta = 90-Vertical.Elevation;
horizSlice = Horizontal.Magnitude;
phi = Horizontal.Azimuth;

[pat3D, thetaout, phiout] = patternFromSlices(vertSlice,theta,horizSlice,phi,Method="CrossWeighted");

save('pat3D_pablo.mat', 'pat3D');

disp('pat3D: ')
disp(pat3D)
disp('thetaout: ')
disp(thetaout)
disp('phiout: ')
disp(phiout)