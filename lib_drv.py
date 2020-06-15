from pololu_drv8835_rpi import motors, MAX_SPEED
import time
import sys

# Set up sequences of motor speeds.
test_forward_speeds = list(range(0, MAX_SPEED, 1)) + \
                      [MAX_SPEED] * 200 + list(range(MAX_SPEED, 0, -1)) + [0]

test_reverse_speeds = list(range(0, -MAX_SPEED, -1)) + \
                      [-MAX_SPEED] * 200 + list(range(-MAX_SPEED, 0, 1)) + [0]

def forward():
    try:
        motors.setSpeeds(0, 0)

        for s in test_forward_speeds:
            motors.setSpeeds(s, s)
            time.sleep(0.005)
        print("Let's go forward")

    finally:
        # Stop the motors, even if there is an exception
        # or the user presses Ctrl+C to kill the process.
        motors.setSpeeds(0, 0)

def backward():
    try:
        motors.setSpeeds(0, 0)

        for s in test_reverse_speeds:
            motors.setSpeeds(s, s)
            time.sleep(0.005)
        print("Let's go backward")

    finally:
        # Stop the motors, even if there is an exception
        # or the user presses Ctrl+C to kill the process.
        motors.setSpeeds(0, 0)

def stop():
    try:
        motors.setSpeeds(0, 0)
        print("Stooop")

    finally:
        # Stop the motors, even if there is an exception
        # or the user presses Ctrl+C to kill the process.
        motors.setSpeeds(0, 0)

def left():
    try:
        motors.setSpeeds(0, 0)
        print("Let's go left")
        for s in test_forward_speeds:
            motors.setSpeeds(s, int(0.25*s))
            time.sleep(0.005)

    finally:
        # Stop the motors, even if there is an exception
        # or the user presses Ctrl+C to kill the process.
        motors.setSpeeds(0, 0)

def right():
    try:
        motors.setSpeeds(0, 0)
        print("Let's go right")

        for s in test_forward_speeds:
            motors.setSpeeds(int(0.25*s), s)
            time.sleep(0.005)

    finally:
        # Stop the motors, even if there is an exception
        # or the user presses Ctrl+C to kill the process.
        motors.setSpeeds(0, 0)

def past():
    pass

def execute(x):
    if x == 'DO_PRZODU':
        return forward()
    elif x == 'DO_TYLU':
        return backward()
    elif x == 'W_LEWO':
        return left()
    elif x == 'W_PRAWO':
        return right()
    elif x == 'STOP':
        return stop()
    else:
        return past()

if __name__ == '__main__':
    label = (sys.argv[1])
    execute(label)
