from gpiozero import DigitalOutputDevice
from time import sleep

# DRV8833 GPIO mapping
# Physical left pump:  GPIO17 / GPIO27
# Physical right pump: GPIO22 / GPIO23

L1 = DigitalOutputDevice(17, initial_value=False)
L2 = DigitalOutputDevice(27, initial_value=False)
R1 = DigitalOutputDevice(22, initial_value=False)
R2 = DigitalOutputDevice(23, initial_value=False)

MAX_SEC = 3.0
GAP_SEC = 0.05
ALT_CYCLES = 4


def stop():
    L1.off()
    L2.off()
    R1.off()
    R2.off()


def get_seconds(prompt, default):
    try:
        text = input(f"{prompt} [{default:.2f}]: ").strip()
        seconds = float(text) if text else default

        if seconds <= 0:
            print("Duration must be greater than 0.")
            return None

        if seconds > MAX_SEC:
            print(f"Maximum allowed duration is {MAX_SEC:.1f} seconds.")
            return None

        return seconds

    except ValueError:
        print("Invalid duration.")
        return None


def left_pump(seconds):
    stop()
    L1.on()
    L2.off()
    sleep(seconds)
    stop()


def right_pump(seconds):
    stop()
    R1.on()
    R2.off()
    sleep(seconds)
    stop()


def turn_left():
    # Turn LEFT = physical RIGHT pump
    seconds = get_seconds("LEFT turn seconds", 0.70)
    if seconds is None:
        return

    print(f">>> TURN LEFT: right pump {seconds:.2f}s <<<")
    right_pump(seconds)
    print(">>> STOPPED <<<")


def turn_right():
    # Turn RIGHT = physical LEFT pump
    seconds = get_seconds("RIGHT turn seconds", 0.70)
    if seconds is None:
        return

    print(f">>> TURN RIGHT: left pump {seconds:.2f}s <<<")
    left_pump(seconds)
    print(">>> STOPPED <<<")


def straight():
    seconds = get_seconds("Both-pump forward seconds", 1.00)
    if seconds is None:
        return

    print(f">>> BOTH PUMPS: {seconds:.2f}s <<<")

    stop()
    L1.on()
    L2.off()
    R1.on()
    R2.off()

    sleep(seconds)
    stop()

    print(">>> STOPPED <<<")


def staggered_straight():
    delay = get_seconds("Delay before second pump", 0.30)
    if delay is None:
        return

    together = get_seconds("Both-pumps running time", 1.00)
    if together is None:
        return

    print(
        f">>> STAGGERED: left first, delay={delay:.2f}s, "
        f"together={together:.2f}s <<<"
    )

    stop()

    # Physical left pump first
    L1.on()
    L2.off()
    sleep(delay)

    # Add physical right pump
    R1.on()
    R2.off()
    sleep(together)

    stop()
    print(">>> STOPPED <<<")


def alternating_straight():
    right_sec = get_seconds("Right pump seconds", 0.25)
    if right_sec is None:
        return

    left_sec = get_seconds("Left pump seconds", 0.25)
    if left_sec is None:
        return

    print(
        f">>> ALTERNATING: {ALT_CYCLES} cycles | "
        f"RIGHT={right_sec:.2f}s | LEFT={left_sec:.2f}s <<<"
    )

    for cycle in range(ALT_CYCLES):

        print(f">>> Cycle {cycle + 1}/{ALT_CYCLES}: RIGHT PUMP <<<")
        right_pump(right_sec)
        sleep(GAP_SEC)

        print(f">>> Cycle {cycle + 1}/{ALT_CYCLES}: LEFT PUMP <<<")
        left_pump(left_sec)
        sleep(GAP_SEC)

    stop()
    print(">>> ALTERNATING COMPLETE - STOPPED <<<")


try:
    stop()

    print("""
=================================================
 OCEAN VOYAGER - MANUAL POOL CALIBRATION
=================================================

a = ALTERNATING FORWARD
    asks right-pump + left-pump seconds

l = TURN LEFT
    asks turn duration

r = TURN RIGHT
    asks turn duration

f = BOTH PUMPS
    asks duration

g = STAGGERED BOTH-PUMP TEST
    asks stagger + together duration

s = STOP
q = QUIT

Maximum individual duration: 3 seconds.
Each movement automatically stops.
=================================================
""")

    while True:

        command = input("Choose [a/l/r/f/g/s/q]: ").strip().lower()

        if command == "a":
            alternating_straight()

        elif command == "l":
            turn_left()

        elif command == "r":
            turn_right()

        elif command == "f":
            straight()

        elif command == "g":
            staggered_straight()

        elif command == "s":
            stop()
            print(">>> STOPPED <<<")

        elif command == "q":
            stop()
            print(">>> QUIT - PUMPS OFF <<<")
            break

        else:
            print("Unknown command.")

finally:
    stop()
    L1.close()
    L2.close()
    R1.close()
    R2.close()
