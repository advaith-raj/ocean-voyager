#!/usr/bin/env python3
"""Template for future real peripheral services.

Build 0.5 writes file-based command/response traces in-process. When moving to
real hardware, this template can be copied into separate services such as:
  - run_real_thermal_service.py
  - run_real_camera_service.py
  - run_real_motor_service.py

The service should read command JSON files from device_bus/commands/pending,
perform real I2C/camera/PCA9685 work, write response JSON files to
responses/pending, and move processed commands into commands/processed.
"""
from __future__ import annotations

print("Template only. Use this as the starting point for real peripheral services.")
