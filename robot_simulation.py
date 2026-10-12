import math
from pathlib import Path

import pygame

# Window and simulation settings
WIDTH, HEIGHT = 1000, 700
FPS = 60
DT = 1.0 / FPS
WHEEL_BASE = 40.0
MAX_SPEED = 160.0
MAX_TURN = 3.0
STOP_DISTANCE = 8.0
MAX_TIME = 30.0

# Distance controller gains
KP_DISTANCE = 0.8
KD_DISTANCE = 0.4

# Heading controller gains
KP_HEADING = 3.0
KI_HEADING = 0.01
KD_HEADING = 0.2
INTEGRAL_LIMIT = 2.0

START = (150.0, 550.0, 0.0)
TARGET = (800.0, 180.0)
SCREENSHOT_DIR = Path(__file__).resolve().parent / "screenshots"


def wrap_angle(angle):
    """Return an equivalent angle in [-pi, pi)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def clamp(value, low, high):
    return max(low, min(high, value))


def calculate_errors(x, y, theta):
    dx = TARGET[0] - x
    dy = TARGET[1] - y
    distance = math.hypot(dx, dy)
    heading = wrap_angle(math.atan2(dy, dx) - theta)
    return distance, heading


def reset():
    """Reset both robot pose and controller memory."""
    x, y, theta = START
    return {
        "x": x,
        "y": y,
        "theta": theta,
        "path": [(x, y)],
        "elapsed": 0.0,
        "previous_distance": None,
        "previous_heading": None,
        "heading_integral": 0.0,
        "status": "MOVING",
    }


def update(robot, mode):
    """Advance the simulation by one fixed timestep."""
    if robot["status"] != "MOVING":
        return

    distance, heading = calculate_errors(
        robot["x"], robot["y"], robot["theta"]
    )

    if distance <= STOP_DISTANCE:
        robot["status"] = "TARGET REACHED"
        return

    if robot["elapsed"] >= MAX_TIME:
        robot["status"] = "TIME LIMIT"
        return

    # No derivative spike on the first update.
    previous_distance = robot["previous_distance"]
    previous_heading = robot["previous_heading"]

    distance_derivative = (
        0.0 if previous_distance is None
        else (distance - previous_distance) / DT
    )
    heading_derivative = (
        0.0 if previous_heading is None
        else wrap_angle(heading - previous_heading) / DT
    )

    # P mode: P distance. PD/PID modes: PD distance.
    v = KP_DISTANCE * distance
    if mode in ("PD", "PID"):
        v += KD_DISTANCE * distance_derivative

    # P and PD modes use P heading.
    omega = KP_HEADING * heading

    # PID mode adds integral and derivative heading action.
    if mode == "PID":
        integral = robot["heading_integral"] + heading * DT
        robot["heading_integral"] = clamp(
            integral, -INTEGRAL_LIMIT, INTEGRAL_LIMIT
        )
        omega += (
            KI_HEADING * robot["heading_integral"]
            + KD_HEADING * heading_derivative
        )

    # Limit speed and reduce forward motion when misaligned.
    alignment = max(0.0, math.cos(heading))
    v = clamp(v * alignment, 0.0, MAX_SPEED)
    omega = clamp(omega, -MAX_TURN, MAX_TURN)

    # Inverse kinematics: body commands to wheel speeds.
    v_left = v - WHEEL_BASE * omega / 2
    v_right = v + WHEEL_BASE * omega / 2

    # Forward kinematics: wheel speeds to body motion.
    body_v = (v_right + v_left) / 2
    body_omega = (v_right - v_left) / WHEEL_BASE

    # Pygame convention: positive y points downward.
    theta = robot["theta"]
    robot["x"] += body_v * math.cos(theta) * DT
    robot["y"] += body_v * math.sin(theta) * DT
    robot["theta"] = wrap_angle(theta + body_omega * DT)

    robot["previous_distance"] = distance
    robot["previous_heading"] = heading
    robot["elapsed"] += DT
    robot["path"].append((robot["x"], robot["y"]))

    # Check arrival using the updated position.
    final_distance, _ = calculate_errors(
        robot["x"], robot["y"], robot["theta"]
    )
    if final_distance <= STOP_DISTANCE:
        robot["status"] = "TARGET REACHED"
    elif robot["elapsed"] >= MAX_TIME:
        robot["status"] = "TIME LIMIT"


def draw(screen, font, robot, mode):
    screen.fill((245, 247, 250))

    for gx in range(0, WIDTH, 50):
        pygame.draw.line(
            screen, (225, 230, 235), (gx, 0), (gx, HEIGHT)
        )
    for gy in range(0, HEIGHT, 50):
        pygame.draw.line(
            screen, (225, 230, 235), (0, gy), (WIDTH, gy)
        )

    if len(robot["path"]) > 1:
        pygame.draw.lines(
            screen, (30, 160, 90), False, robot["path"], 3
        )

    pygame.draw.circle(
        screen, (110, 110, 110),
        (round(START[0]), round(START[1])), 6
    )

    x, y, theta = robot["x"], robot["y"], robot["theta"]
    center = (round(x), round(y))
    pygame.draw.circle(screen, (40, 110, 220), center, 18)

    tip = (
        round(x + 32 * math.cos(theta)),
        round(y + 32 * math.sin(theta)),
    )
    pygame.draw.line(screen, (245, 145, 30), center, tip, 5)
    pygame.draw.circle(screen, (245, 145, 30), tip, 4)

    if robot["status"] == "TARGET REACHED":
        pygame.draw.circle(
            screen, (120, 50, 180), center, 24, 2
        )

    # Draw a target cross above the robot so it remains visible.
    tx, ty = TARGET
    pygame.draw.circle(screen, (210, 50, 50), TARGET, 12, 2)
    pygame.draw.line(
        screen, (210, 50, 50), (tx - 6, ty), (tx + 6, ty), 2
    )
    pygame.draw.line(
        screen, (210, 50, 50), (tx, ty - 6), (tx, ty + 6), 2
    )

    distance, heading = calculate_errors(x, y, theta)
    info = [
        f"Controller: {mode}",
        f"Distance error: {distance:.2f} px",
        f"Heading error: {math.degrees(heading):.2f} degrees",
        f"Simulation time: {robot['elapsed']:.2f} s",
        f"Status: {robot['status']}",
        "1: P | 2: PD | 3: PID",
        "R: reset | S: save screenshot | ESC: exit",
    ]

    pygame.draw.rect(
        screen, (255, 255, 255), (10, 10, 480, 220)
    )
    for index, text in enumerate(info):
        screen.blit(
            font.render(text, True, (35, 40, 50)),
            (20, 20 + index * 28),
        )


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 22)

    mode = "PID"
    robot = reset()
    running = True

    while running:
        clock.tick(FPS)
        save_requested = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    robot = reset()
                elif event.key in (
                    pygame.K_1, pygame.K_2, pygame.K_3
                ):
                    mode = {
                        pygame.K_1: "P",
                        pygame.K_2: "PD",
                        pygame.K_3: "PID",
                    }[event.key]
                    robot = reset()
                elif event.key == pygame.K_s:
                    save_requested = True

        if not running:
            break

        pygame.display.set_caption(
            f"Differential Drive Robot - {mode} Control"
        )
        update(robot, mode)
        draw(screen, font, robot, mode)
        pygame.display.flip()

        if save_requested:
            SCREENSHOT_DIR.mkdir(exist_ok=True)
            filename = SCREENSHOT_DIR / f"{mode}_result.png"
            pygame.image.save(screen, str(filename))
            print(f"Screenshot saved: {filename}")

    pygame.quit()


if __name__ == "__main__":
    main()