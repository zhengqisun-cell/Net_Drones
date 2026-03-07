import os
import time

import mujoco


HERE = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(HERE, "net_drone_fishing.xml")
PRINT_SIM_TIME_EVERY = 0.5  # seconds (wall clock)


def main():
    if not os.path.exists(XML_PATH):
        raise FileNotFoundError(f"XML not found: {XML_PATH}")

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    try:
        import mujoco.viewer as viewer
    except Exception as exc:
        raise RuntimeError(
            "mujoco.viewer is unavailable. Please install a MuJoCo build with viewer support."
        ) from exc

    with viewer.launch_passive(model, data) as v:
        # A slightly pulled-back default camera for quick structure inspection.
        v.cam.distance = 9.0
        v.cam.elevation = -20.0
        v.cam.azimuth = 120.0

        last_print_wall = time.time()
        print("Viewer started. Press ESC in window to quit.")
        print("If structure looks static, check console sim_time below.")

        while v.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            v.sync()

            now = time.time()
            if (now - last_print_wall) >= PRINT_SIM_TIME_EVERY:
                last_print_wall = now
                print(f"sim_time={data.time:.3f}s")

            dt_sleep = float(model.opt.timestep) - (time.time() - step_start)
            if dt_sleep > 0.0:
                time.sleep(dt_sleep)


if __name__ == "__main__":
    main()
