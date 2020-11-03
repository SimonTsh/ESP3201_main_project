"""manual_logged_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Camera, Display, Keyboard
from vehicle import Driver
import csv

# get the time step of the current world.
timestep = 50
# timestep = int(robot.getBasicTimeStep())

# create the Robot instance.
driver = Driver()
kb = Keyboard()
kb.enable(timestep)

speed = 25.0
steering_angle = 0.0
manual_steering = 0
count = 0

driver.setCruisingSpeed(speed)

def set_speed(kmh):
    speed = min(kmh, 250.0);
    print("setting speed to ", speed, " km/h");
    driver.setCruisingSpeed(speed);
    return speed

def set_steering_angle(steering_angle, wheel_angle):
    if (wheel_angle - steering_angle > 0.1):
        wheel_angle = steering_angle + 0.1;
    if (wheel_angle - steering_angle < -0.1):
        wheel_angle = steering_angle - 0.1;
    steering_angle = wheel_angle;

    if (wheel_angle > 0.5):
        wheel_angle = 0.5;
    elif (wheel_angle < -0.5):
        wheel_angle = -0.5;
    driver.setSteeringAngle(wheel_angle);
    return steering_angle


def change_manual_steer_angle(curr_angle, curr_steering, inc):
    new_manual_steering = curr_steering + inc;
    if ((new_manual_steering <= 25) & (new_manual_steering >= -25)):
        curr_steering = new_manual_steering
        steering_angle = set_steering_angle(curr_angle, curr_steering * 0.02);

    if (manual_steering == 0):
        print("going straight")
    else:
        print("turning ", steering_angle, "%.2f rad (", "left" if (steering_angle < 0) else "right", ")")

    return (curr_steering, steering_angle)


def checkKeyboard(speed, manual_steering, steering_angle):
    key = kb.getKey()
    angle = 0.0
    if (key == ord('W')):
        speed = set_speed(speed + 0.8)
    elif (key == ord('S')):
        speed = set_speed(speed - 2.0)
    elif (key == ord('D')):
        # (manual_steering, steering_angle) = change_manual_steer_angle(steering_angle, manual_steering, 1)
        angle += 0.2
    elif (key == ord('A')):
        angle -= 0.2
    driver.setSteeringAngle(angle)
    steering_angle = angle
        # (manual_steering, steering_angle) = change_manual_steer_angle(steering_angle, manual_steering, -1)

    return (speed, manual_steering, steering_angle)




# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getMotor('motorname')
#  ds = robot.getDistanceSensor('dsname')
#  ds.enable(timestep)

# Main loop:
# - perform simulation steps until Webots is stopping the controller



with open('drive_data.csv', mode='w', newline = '') as drive_data:
    writer = csv.DictWriter(drive_data, fieldnames=['idx','speed','angle'])
    writer.writeheader()

    while driver.step() != -1:
        (speed, manual_steering, steering_angle) = checkKeyboard(speed, manual_steering, steering_angle)
        # Read the sensors:
        # Enter here functions to read sensor data, like:
        #  val = ds.getValue()
        write_to_log = writer.writerow({'idx': count, 'speed': speed, 'angle': steering_angle})
        count += 1

        # Process sensor data here.

        # Enter here functions to send actuator commands, like:
        #  motor.setPosition(10.0)
        pass

    # Enter here exit cleanup code.
