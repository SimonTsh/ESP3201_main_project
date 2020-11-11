%% read data

T = readtable('comp_filter_data.csv');
eulers = table2array([T(:,'roll'), T(:,'pitch'), T(:,'yaw')]);
gyro = table2array([T(:,'gyro_x'),T(:,'gyro_y'),T(:,'gyro_z')]);
accel = table2array([T(:,'acc_x'),T(:,'acc_y'),T(:,'acc_z')]);
dt = 0.01;

%% computations (euler)

angles_gyro = cumsum(gyro,1) * dt;
roll_acc = atan2(-accel(:,3), accel(:,2));
pitch_acc = atan2(accel(:,1), sqrt(accel(:,2).^2 + accel(:,3).^2));

% figure(1)
% h = plot(angles_gyro(:,1));     % roll
% hold on
% h(2) = plot(angles_gyro(:,2));  % yaw
% h(3) = plot(angles_gyro(:,3));  % pitch
% legend("Roll","Yaw","Pitch")

% figure(2)
% g = plot(roll_acc); hold on
% g(2) = plot(pitch_acc);

% figure(3)
% h = plot(eulers(:,1));     % roll
% hold on
% h(2) = plot(eulers(:,3));  % yaw
% h(3) = plot(eulers(:,2));  % pitch
% legend("Roll","Yaw","Pitch")

ALPHA = 0.999999;
filter = zeros(length(eulers),2);   % ignore yaw; [pitch roll]
filter(1,:) = [0.0, 0.0];

for i=2:length(eulers)
    filter(i,:) = ALPHA * (filter(i-1,:) + dt * gyro(i-1,[3 1])) + (1 - ALPHA) * [pitch_acc(i-1), roll_acc(i-1)];
end

figure(4)
subplot(2,1,1)
plot(filter(:,1)); title('pitch'); hold on
plot(eulers(:,2)); legend("FILTER","EULER")
hold off
subplot(2,1,2)
plot(rad2deg(filter(:,2))); title('roll'); hold on
plot(rad2deg(eulers(:,1))); legend("FILTER","EULER")
hold off

%% Computations (quat)

ALPHA = 0.9999;
filter_q = zeros(length(eulers),4);
filter_q(1,:) = [1,0,0,0];
filter = zeros(length(eulers),3);

for i=2:length(eulers)
qDelta = rotationVecToQuat(gyro(i-1,:)*dt);
qw = quatMultiply(filter_q(i-1,:), qDelta);

% qa_world = [0; rotationQuatToMatrix(qw) * accel(i-1,:)];
v = vecNormalize(rotationQuatToMatrix(qw) * accel(i-1,:)');
qa = rotationVecToQuat((1 - ALPHA) * acos(v(2)) * vecNormalize([-v(3), 0, v(1)]'));
filter_q(i,:) = quatMultiply(qa, qw);
filter(i,:) = quatToEul(filter_q(i,:));
end


figure(4)
subplot(2,1,1)
plot(filter(:,3)); title('pitch'); hold on
plot(eulers(:,2)); legend("FILTER","EULER")
hold off
subplot(2,1,2)
plot(filter(:,3)); title('roll'); hold on
plot(eulers(:,1)); legend("FILTER","EULER")
hold off

%% functions

function vhat = vecNormalize(v)
vecnorm = sqrt(sum(v.^2));
if vecnorm < 1e-6
    vhat = v / (vecnorm + 1e-6);
else
    vhat = v / vecnorm;
end
end

function qInv = quatInverse(q)
qInv = q;
qInv(2:4) = -q(2:4);
qInv = qInv ./ sum(q.^2);
end

function pq = quatMultiply(p, q)
% Returns a quaternion which is the result of the quaternion product p(*)q.
% Inputs: 
% p - a [4 x 1] quaternion
% q - a [4 x 1] quaternion
% Output: 
% pq - a new [4 x 1] quaternion
pq = zeros(4,1);
pq(1) = p(1)*q(1) - p(2)*q(2) - p(3)*q(3) - p(4)*q(4);
pq(2) = p(1)*q(2) + p(2)*q(1) + p(3)*q(4) - p(4)*q(3);
pq(3) = p(1)*q(3) - p(2)*q(4) + p(3)*q(1) + p(4)*q(2);
pq(4) = p(1)*q(4) + p(2)*q(3) - p(3)*q(2) + p(4)*q(1);
end

function q = rotationVecToQuat(v)
% Returns an orientation quaternion corresponding to a rotation by the
% vector v = |v|*u, where u is the axis of rotation and |v| is the rotated
% angle.
% Input: 
% v - a [3 x 1] rotation vector
% Output: 
% q - the [4 x 1] quaternion corresponding to a rotation by v

vecnorm = sqrt(sum(v.^2));
q = zeros(4,1);
% small argument limit
if vecnorm <= 1e-6
    q(1) = 1;
    q(2:4) = v/2;
    return
end
q(1) = cos(vecnorm * 0.5);
q(2:4) = v/vecnorm * sin(vecnorm * 0.5);
end

function M = vecToSkew(v)
% Returns the skew-symmetric matrix corresponding to the action of applying
% cross(v,u) onto a vector u.
% Input: 
% v - a [3 x 1] column vector
% Output: 
% M - a [3 x 3] skew matrix
M = [   0, -v(3),  v(2);
    v(3),   0,  -v(1);
    -v(2),  v(1),   0];
end

function R = rotationQuatToMatrix(q)
% Converts an orientation quaternion into its corresponding rotation
% matrix. The quaternion should be a valid orientation quaternion (unit
% length); otherwise, it is normalized inside the function.
%
% Input: 
% q - a [4 x 1] unit orientation quaternion
% Output: 
% R - a [3 x 3] rotation matrix corresponding to q
qlengthSq = sum(q.^2);
if qlengthSq >= 1.001
   q = q / sqrt(qlengthSq);
end
R = zeros(3);
R(1,1) = q(1)*q(1) + q(2)*q(2) - q(3)*q(3) - q(4)*q(4);
R(1,2) = 2*(q(2)*q(3) - q(1)*q(4));
R(1,3) = 2*(q(2)*q(4) + q(1)*q(3));
R(2,1) = 2*(q(2)*q(3) + q(1)*q(4));
R(2,2) = q(1)*q(1) - q(2)*q(2) + q(3)*q(3) - q(4)*q(4);
R(2,3) = 2*(q(3)*q(4) - q(1)*q(2));
R(3,1) = 2*(q(2)*q(4) - q(1)*q(3));
R(3,2) = 2*(q(3)*q(4) + q(1)*q(2));
R(3,3) = q(1)*q(1) - q(2)*q(2) - q(3)*q(3) + q(4)*q(4);

% Validation test case
% x = randn(4,1); y = randn(3,1);
% m = quatrotate(quatconj(x'),y')' - rotationQuatToMatrix(x)*y;
% fprintf("%f\n", m);
end

function R = rotationVecToMatrix(v)
vecnorm = sqrt(sum(v.^2));
% small argument limit
if vecnorm <= 1e-6
    skewMat = vecToSkew(v);
    R = eye(3) + skewMat + 0.5*skewMat*skewMat;
    return
end
skewMat = vecToSkew(v./vecnorm);
R = eye(3) + sin(vecnorm) * skewMat + (1 - cos(vecnorm)) * skewMat * skewMat;
end

function angles = quatToEul(q)
    angles = [0 0 0];
    angles(1) = atan2(2*(q(4)*q(1)+q(2)*q(3)),   1-2*(q(3)^2+q(4)^2));
    angles(2) = asin(2*(q(3)*q(1)-q(2)*q(4)));
    angles(3) = atan2(2*(q(1)*q(2)+q(3)*q(4)),   1-2*(q(2)^2+q(3)^2));
end
