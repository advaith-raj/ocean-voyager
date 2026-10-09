from gpiozero import DigitalOutputDevice
from time import sleep

# DRV8833 control mapping
# Left pump:  IN1=GPIO17, IN2=GPIO27
# Right pump: IN3=GPIO22, IN4=GPIO23

L1 = DigitalOutputDevice(17, initial_value=False)
L2 = DigitalOutputDevice(27, initial_value=False)
R1 = DigitalOutputDevice(22, initial_value=False)
R2 = DigitalOutputDevice(23, initial_value=False)

def stop():
    L1.off()
    L2.off()
    R1.off()
    R2.off()

def left_forward():
    L1.on()
    L2.off()

def right_forward():
    R1.on()
    R2.off()

def both_forward():
    L1.on()
    L2.off()
    R1.on()
    R2.off()

stop()

print("GPIO initialized.")
print("All DRV8833 inputs are LOW.")
print("Pumps should be OFF.")
print("Ctrl+C to exit.")

try:
    while True:
        sleep(1)
except KeyboardInterrupt:
    pass
finally:
    stop()
    L1.close()
    L2.close()
    R1.close()
    R2.close()
    print("\nGPIO released; pumps OFF.")
