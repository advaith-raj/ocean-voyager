from gpiozero import DigitalOutputDevice
from time import sleep

# DRV8833 GPIO mapping
# Left:  GPIO17 / GPIO27
# Right: GPIO22 / GPIO23

L1 = DigitalOutputDevice(17, initial_value=False)
L2 = DigitalOutputDevice(27, initial_value=False)
R1 = DigitalOutputDevice(22, initial_value=False)
R2 = DigitalOutputDevice(23, initial_value=False)

def stop():
    L1.off()
    L2.off()
    R1.off()
    R2.off()

def left():
    L1.on()
    L2.off()
    R1.off()
    R2.off()

def right():
    L1.off()
    L2.off()
    R1.on()
    R2.off()

def both():
    L1.on()
    L2.off()
    R1.on()
    R2.off()

def run_test(name, function):
    print(f"\n>>> {name} ON - 20 SECONDS <<<", flush=True)
    function()
    sleep(20)
    stop()
    print(">>> ALL PUMPS OFF - 10 SECONDS <<<", flush=True)
    sleep(10)

try:
    stop()

    print("\nALL PUMPS OFF")
    print("120 SECONDS TO PLACE VOYAGER IN BATHTUB")
    print("Hold Voyager lightly and keep electronics dry.\n")

    for remaining in range(120, 0, -1):
        if remaining in (120, 90, 60, 30, 20, 10, 5, 4, 3, 2, 1):
            print(f"Starting in {remaining} seconds...", flush=True)
        sleep(1)

    print("\n========== TRIAL 1 ==========", flush=True)
    run_test("LEFT PUMP", left)
    run_test("RIGHT PUMP", right)
    run_test("BOTH PUMPS", both)

    print("\n========== TRIAL 2 ==========", flush=True)
    run_test("LEFT PUMP", left)
    run_test("RIGHT PUMP", right)

    # Last test doesn't need the extra 10-second wait
    print("\n>>> BOTH PUMPS ON - 20 SECONDS <<<", flush=True)
    both()
    sleep(20)
    stop()

    print("\n========== TEST COMPLETE ==========", flush=True)
    print("ALL PUMPS OFF", flush=True)

except KeyboardInterrupt:
    print("\nTEST INTERRUPTED", flush=True)

finally:
    stop()
    L1.close()
    L2.close()
    R1.close()
    R2.close()
    print("GPIO released. Pumps OFF.", flush=True)
