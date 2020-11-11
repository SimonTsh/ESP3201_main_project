"""comp_filter_test controller."""

from controller import Accelerometer, Gyro, InertialUnit, Keyboard
from vehicle import Driver
import csv

dt = 50

driver = Driver()
kb = Keyboard()
kb.enable(dt)

accel = driver.getAccelerometer('accel')
gyro = driver.getGyro('gyro')
imu = driver.getInertialUnit('imu')
accel.enable(dt)
gyro.enable(dt)
imu.enable(dt)

speed = 25.0
steering_angle = 0.0


def set_speed(kmh):
    speed = min(kmh, 250.0);
    print("setting speed to ", speed, " km/h");
    driver.setCruisingSpeed(speed);
    return speed

def checkKeyboard(speed, steering_angle):
    key = kb.getKey()
    angle = 0.0
    if (key == ord('W')):
        speed = set_speed(speed + 0.8)
    elif (key == ord('S')):
        speed = set_speed(speed - 2.0)
    elif (key == ord('D')):
        angle += 0.2
    elif (key == ord('A')):
        angle -= 0.2
    driver.setSteeringAngle(angle)
    steering_angle = angle
    return (speed, steering_angle)


names = ['speed','angle','acc-x','acc-y','acc-z','gyro-x','gyro-y','gyro-z','roll','pitch','yaw']

with open('drive_data.csv', mode='w', newline = '') as drive_data:
    writer = csv.DictWriter(drive_data, fieldnames=names)
    writer.writeheader()

    while driver.step() != -1:
        (speed, steering_angle) = checkKeyboard(speed, steering_angle)
    
        acc_vec = accel.getValues()
        gyro_vec= gyro.getValues()
        angle_vec = imu.getRollPitchYaw()
        
        # print("---------")
        # print("R: %.3f" % angle_vec[0])
        # print("P: %.3f" % angle_vec[1])
        # print("Y: %.3f" % angle_vec[2])
    
        write_to_log = writer.writerow({'speed': speed,
                                        'angle': steering_angle,
                                        'acc-x': acc_vec[0],
                                        'acc-y': acc_vec[1],
                                        'acc-z': acc_vec[2],
                                        'gyro-x': gyro_vec[0],
                                        'gyro-y': gyro_vec[1],
                                        'gyro-z': gyro_vec[2],
                                        'roll': angle_vec[0],
                                        'pitch': angle_vec[1],
                                        'yaw': angle_vec[2] })
    

    pass


