#!/usr/bin/env python3
"""Root CLI shim — trajectory dump for Godot. Prefer: python -m collision_avoidance.demo.dump_trajectory"""

from collision_avoidance.demo.dump_trajectory import main

if __name__ == "__main__":
    raise SystemExit(main())
