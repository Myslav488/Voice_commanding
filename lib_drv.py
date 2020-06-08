from pololu_drv8835_rpi import motors, MAX_SPEED
import time

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

    finally:
        # Stop the motors, even if there is an exception
        # or the user presses Ctrl+C to kill the process.
        motors.setSpeeds(0, 0)

def stop():
    try:
        motors.setSpeeds(0, 0)

    finally:
        # Stop the motors, even if there is an exception
        # or the user presses Ctrl+C to kill the process.
        motors.setSpeeds(0, 0)

def left():
    try:
        motors.setSpeeds(0, 0)

        for s in test_forward_speeds:
            motors.setSpeeds(s, 0.25*s)
            time.sleep(0.005)

    finally:
        # Stop the motors, even if there is an exception
        # or the user presses Ctrl+C to kill the process.
        motors.setSpeeds(0, 0)

def right():
    try:
        motors.setSpeeds(0, 0)

        for s in test_forward_speeds:
            motors.setSpeeds(0.25*s, s)
            time.sleep(0.005)

    finally:
        # Stop the motors, even if there is an exception
        # or the user presses Ctrl+C to kill the process.
        motors.setSpeeds(0, 0)

def past():
    pass

def execute(x):
    return {
        'DO_PRZODU': forward(),
        'DO_TYŁU': backward(),
        'W_LEWO': left(),
        'W_PRAWO': right(),
        'STOP': stop(),
        'else': past(),
    }.get(x, 'else')

'''def forward():

    motors.setSpeeds(0, 0)
    for s in test_forward_speeds:
        motors.setSpeeds(s, s)
        time.sleep(0.005)

    motors.setSpeeds(0, 0)

def backward():
    motors.setSpeeds(0, 0)
    for s in test_reverse_speeds:
        motors.setSpeeds(s, s)
        time.sleep(0.005)

    motors.setSpeeds(0, 0)

def stop():
    motors.setSpeeds(0, 0)

def left():
    motors.setSpeeds(0, 0)
    for s in test_forward_speeds:
        motors.setSpeeds(s, 0.25*s)
        time.sleep(0.005)

    motors.setSpeeds(0, 0)

def right():
    motors.setSpeeds(0, 0)
    for s in test_forward_speeds:
        motors.setSpeeds(0.25*s, s)
        time.sleep(0.005)
    motors.setSpeeds(0, 0)

def past():
    print("Inne nic ")
    pass

def execute(x):
    return {
        'DO_PRZODU': forward(),
        'DO_TYŁU': backward(),
        'W_LEWO': left(),
        'W_PRAWO': right(),
        'STOP': stop(),
        'else': past(),
    }.get(x, 'else')'''