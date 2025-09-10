#!/usr/bin/env python3
"""
Playable top-down driving prototype (Pygame).
Controls:
  - Arrow keys or WASD: accelerate / brake / steer
  - SPACE: handbrake (quick slow)
  - M: toggle minimap
  - F5: save player position to save.json
  - F9: load player position from save.json
  - N: skip to next mission / restart current mission
  - ESC or window close: quit

Features:
  - Player car with simple physics (acceleration, steering, friction)
  - NPC traffic that follows waypoint loops
  - Camera that follows player on a larger map
  - Minimap showing player and NPCs
  - Multiple mission types: reach point, collect items, timed race, tail an NPC
  - Mission manager with progression and UI
"""
import sys
import math
import json
import random
import pygame
from pygame.math import Vector2

# === Config ===
SCREEN_W, SCREEN_H = 1280, 720
MAP_W, MAP_H = 2048, 2048
FPS = 60

PLAYER_COLOR = (20, 160, 240)
NPC_COLOR = (220, 80, 60)
ROAD_COLOR = (50, 50, 50)
GRASS_COLOR = (100, 180, 90)
MINIMAP_BG = (20, 20, 20)

SAVE_FILE = "save.json"

# === Utils ===
def clamp(x, a, b):
    return max(a, min(b, x))

def angle_lerp(a, b, t):
    """Shortest-path lerp for angles in degrees."""
    diff = ((b - a + 180) % 360) - 180
    return a + diff * t

# === Car class ===
class Car:
    def __init__(self, pos, angle=0.0, color=(255,255,255), max_speed=300, length=36, width=18):
        self.pos = Vector2(pos)
        self.vel = Vector2(0, 0)
        self.angle = angle  # degrees: 0 points to right
        self.length = length
        self.width = width
        self.color = color
        self.max_speed = max_speed
        self.acceleration = 220.0
        self.brake_accel = 400.0
        self.handbrake_decel = 1200.0
        self.turn_speed = 120.0  # degrees per second at full input
        self.friction = 0.98
        self.bounding_radius = max(self.length, self.width) * 0.6
        # create car surface
        surf = pygame.Surface((self.length, self.width), pygame.SRCALPHA)
        surf.fill(self.color)
        pygame.draw.rect(surf, (0,0,0), surf.get_rect(), 2)
        # draw a simple windshield for orientation cue
        pygame.draw.rect(surf, (30,30,30), (self.length*0.55, self.width*0.15, self.length*0.25, self.width*0.7))
        self.base_surf = surf

    def forward_vector(self):
        rad = math.radians(self.angle)
        return Vector2(math.cos(rad), math.sin(rad))

    def update(self, dt, throttle=0.0, brake=0.0, steer=0.0, handbrake=False):
        # throttle: -1..1 (negative for reverse)
        # brake: 0..1
        # steer: -1..1 left/right
        fwd = self.forward_vector()
        if throttle > 0:
            self.vel += fwd * (self.acceleration * throttle * dt)
        elif throttle < 0:
            self.vel += fwd * (self.acceleration * throttle * dt * 0.6)

        if brake > 0:
            # braking reduces speed in facing direction
            self.vel -= self.vel.normalize() * (self.brake_accel * brake * dt) if self.vel.length() > 1e-3 else Vector2(0,0)

        if handbrake:
            # heavy deceleration
            self.vel *= clamp(1 - self.handbrake_decel * dt / max(self.max_speed,1), 0.0, 1.0)

        # limit speed
        if self.vel.length() > self.max_speed:
            self.vel.scale_to_length(self.max_speed)

        # steering effect scales with forward speed sign and magnitude
        speed_factor = clamp(self.vel.length() / max(1.0, self.max_speed), 0.0, 1.0)
        turn_amount = steer * self.turn_speed * speed_factor * dt * (1 if self.vel.dot(fwd) >= 0 else -1)
        self.angle = (self.angle + turn_amount) % 360

        # apply friction
        self.vel *= self.friction ** dt

        # integrate
        self.pos += self.vel * dt

    def draw(self, surface, camera_offset):
        screen_pos = Vector2(self.pos) - camera_offset
        rotated = pygame.transform.rotate(self.base_surf, -self.angle)
        rect = rotated.get_rect(center=(int(screen_pos.x), int(screen_pos.y)))
        surface.blit(rotated, rect.topleft)

    def rect(self):
        # Approximate bounding rect from center and size (axis-aligned)
        r = pygame.Rect(0,0,self.length, self.width)
        r.center = (self.pos.x, self.pos.y)
        return r

# === NPC driver ===
class NPC:
    def __init__(self, path_points, start_index=0, speed=140):
        self.path = [Vector2(p) for p in path_points]
        self.index = start_index % len(self.path)
        self.car = Car(self.path[self.index], angle=0, color=NPC_COLOR, max_speed=speed)
        # set next target
        self.target_index = (self.index + 1) % len(self.path)
        self.tolerance = 12.0

    def update(self, dt):
        if not self.path:
            return
        target = self.path[self.target_index]
        to_target = target - self.car.pos
        dist = to_target.length()
        desired_angle = math.degrees(math.atan2(to_target.y, to_target.x)) if dist>1e-3 else self.car.angle
        angle_diff = ((desired_angle - self.car.angle + 180) % 360) - 180
        steer = clamp(angle_diff / 45.0, -1, 1)
        throttle = 1.0 if dist > 120 else 0.6 if dist > 40 else 0.2
        # slow down when sharp turn needed
        if abs(angle_diff) > 60:
            throttle *= 0.4
        self.car.update(dt, throttle=throttle, steer=steer, handbrake=False)
        if dist < self.tolerance:
            # move to next waypoint
            self.target_index = (self.target_index + 1) % len(self.path)

    def draw(self, surface, camera_offset):
        self.car.draw(surface, camera_offset)

# === Camera ===
class Camera:
    def __init__(self, width, height, map_w, map_h):
        self.w = width
        self.h = height
        self.map_w = map_w
        self.map_h = map_h
        self.pos = Vector2(width//2, height//2)

    def follow(self, target_pos):
        halfw, halfh = self.w/2, self.h/2
        x = clamp(target_pos.x - halfw, 0, max(0, self.map_w - self.w))
        y = clamp(target_pos.y - halfh, 0, max(0, self.map_h - self.h))
        self.pos = Vector2(x, y)

# === Map & roads (simple grid with waypoints) ===
def generate_waypoints_grid(cell=256, offset=64):
    waypoints = []
    # horizontal streets
    for y in range(offset, MAP_H, cell):
        row = [(x, y) for x in range(offset, MAP_W, cell)]
        waypoints.append(row)
    # flatten and also add vertical traversals for NPC loops
    flattened = []
    # create loops that follow a rectangle around center blocks
    loops = []
    for i in range(2, 6):
        margin = i * cell
        loop = [
            (margin, margin),
            (MAP_W - margin, margin),
            (MAP_W - margin, MAP_H - margin),
            (margin, MAP_H - margin),
        ]
        loops.append(loop)
    # plus some cross-town diagonal loops
    loops.append([(200,200),(400,600),(800,400),(1200,900),(1800,600),(1500,300)])
    return loops

# === Mission system ===
class Mission:
    def __init__(self, mtype, title, **kwargs):
        self.type = mtype  # 'reach', 'collect', 'timed', 'tail'
        self.title = title
        self.params = kwargs
        self.active = False
        self.completed = False
        self.start_time = 0.0
        self.progress = 0

    def start(self, now):
        self.active = True
        self.completed = False
        self.start_time = now
        self.progress = 0
        # initialize collect items if needed
        if self.type == 'collect':
            # items are list of positions in params['items'] already
            self.params['collected'] = [False] * len(self.params.get('items', []))
        if self.type == 'tail':
            self.params['tail_time'] = 0.0

    def update(self, now, player, npcs):
        if not self.active or self.completed:
            return
        if self.type == 'reach':
            target = Vector2(self.params['point'])
            if (player.pos - target).length() < self.params.get('radius', 40):
                self.completed = True
        elif self.type == 'collect':
            items = self.params.get('items', [])
            for i, p in enumerate(items):
                if not self.params['collected'][i] and (player.pos - Vector2(p)).length() < self.params.get('radius', 28):
                    self.params['collected'][i] = True
                    self.progress += 1
            if all(self.params['collected']):
                self.completed = True
        elif self.type == 'timed':
            # must reach finish within time_limit seconds from start
            time_left = self.params.get('time_limit', 20) - (now - self.start_time)
            target = Vector2(self.params['point'])
            if time_left <= 0:
                self.completed = False
                self.active = False  # failed, mark inactive so player can retry or skip
            elif (player.pos - target).length() < self.params.get('radius', 36):
                self.completed = True
        elif self.type == 'tail':
            # follow a target NPC and keep within distance for duration
            npc = self.params.get('npc')
            if npc is None:
                return
            dist = (player.pos - npc.car.pos).length()
            maxd = self.params.get('max_distance', 120)
            if dist <= maxd:
                self.params['tail_time'] += now - self.start_time if self.start_time else 0
                # cap but accumulate correctly using last update pattern handled below

    def tail_update_dt(self, dt, player, npcs):
        if not self.active or self.completed or self.type != 'tail':
            return
        npc = self.params.get('npc')
        if npc is None:
            return
        dist = (player.pos - npc.car.pos).length()
        if dist <= self.params.get('max_distance', 120):
            self.params['tail_time'] += dt
        else:
            # losing contact reduces progress slightly
            self.params['tail_time'] = max(0.0, self.params['tail_time'] - dt*0.5)
        if self.params['tail_time'] >= self.params.get('required_time', 8.0):
            self.completed = True

# === Rendering utilities ===
def draw_map(surface):
    surface.fill(GRASS_COLOR)
    # draw a simple grid of roads
    road_w = 64
    # vertical roads
    for x in range(128, MAP_W, 256):
        pygame.draw.rect(surface, ROAD_COLOR, (x - road_w//2, 0, road_w, MAP_H))
    # horizontal roads
    for y in range(128, MAP_H, 256):
        pygame.draw.rect(surface, ROAD_COLOR, (0, y - road_w//2, MAP_W, road_w))
    # intersections can be slightly darker
    for x in range(128, MAP_W, 256):
        for y in range(128, MAP_H, 256):
            pygame.draw.circle(surface, (40,40,40), (x,y), 36)

# === Main ===
def main():
    pygame.init()
    pygame.display.set_caption("Pygame Open-World Starter Prototype")
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    clock = pygame.time.Clock()

    # large map surface for simple demo (not memory-optimal but okay for prototype)
    world = pygame.Surface((MAP_W, MAP_H))
    draw_map(world)

    # generate waypoint loops and NPCs
    waypoint_loops = generate_waypoints_grid()
    npcs = []
    for loop in waypoint_loops:
        # spawn an NPC per loop (random start index)
        if len(loop) < 2:
            continue
        start = random.randrange(len(loop))
        npc = NPC(loop, start_index=start, speed=random.randint(100, 180))
        # evenly offset the NPC position a bit so they don't all overlap the road center
        offset = Vector2(random.uniform(-8,8), random.uniform(-8,8))
        npc.car.pos += offset
        npcs.append(npc)

    # player car start in center
    player = Car((MAP_W//2, MAP_H//2), angle=0, color=PLAYER_COLOR, max_speed=360)
    camera = Camera(SCREEN_W, SCREEN_H, MAP_W, MAP_H)

    show_minimap = True

    font = pygame.font.SysFont(None, 24)
    bigfont = pygame.font.SysFont(None, 42)

    # Setup missions
    missions = []
    now = 0.0
    # reach mission
    missions.append(Mission('reach', 'Drive to the yellow marker', point=(MAP_W//2 + 420, MAP_H//2 + 200), radius=36))
    # collect mission: scatter 4 items near center
    collect_items = [ (MAP_W//2 + dx, MAP_H//2 + dy) for dx,dy in [(-200,-40),(220,-80),(80,180),(-60,240)] ]
    missions.append(Mission('collect', 'Collect the 4 packages', items=collect_items, radius=28))
    # timed race: start near center, finish point further out, time limit
    missions.append(Mission('timed', 'Timed dash: reach the finish in time', point=(MAP_W//2 + 900, MAP_H//2 - 300), time_limit=18, radius=36))
    # tail NPC: follow one of the npcs for some seconds
    tail_target = npcs[0] if npcs else None
    missions.append(Mission('tail', 'Tail the red car: stay close for 8s', npc=tail_target, max_distance=140, required_time=8.0))

    current_mission = 0
    if missions:
        missions[current_mission].start(now)

    paused = False

    time_acc = 0.0
    while True:
        dt = clock.tick(FPS) / 1000.0
        time_acc += dt
        now = time_acc
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return
                if event.key == pygame.K_m:
                    show_minimap = not show_minimap
                if event.key == pygame.K_n:
                    # skip to next mission or restart current if failed/inactive
                    if missions:
                        # advance index
                        current_mission = (current_mission + 1) % len(missions)
                        missions[current_mission].start(now)
                if event.key == pygame.K_F5:
                    # save player state and mission index
                    data = {"player_pos": [player.pos.x, player.pos.y], "player_angle": player.angle, "mission_index": current_mission}
                    with open(SAVE_FILE, "w") as f:
                        json.dump(data, f)
                    print("Saved.")
                if event.key == pygame.K_F9:
                    try:
                        with open(SAVE_FILE, "r") as f:
                            data = json.load(f)
                            player.pos = Vector2(data.get("player_pos", [player.pos.x, player.pos.y]))
                            player.angle = data.get("player_angle", player.angle)
                            current_mission = data.get("mission_index", current_mission)
                            # restart mission state
                            for i, m in enumerate(missions):
                                m.active = False
                                m.completed = False
                            if missions:
                                missions[current_mission].start(now)
                        print("Loaded.")
                    except Exception as e:
                        print("Failed to load:", e)

        keys = pygame.key.get_pressed()
        # input mapping
        accel = 0.0
        brake = 0.0
        steer = 0.0
        handbrake = keys[pygame.K_SPACE]
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            accel = 1.0
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            accel = -0.6  # reverse
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            steer = -1.0
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            steer = 1.0
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            # holding shift acts as a brake modifier
            brake = 0.8

        # Update NPCs
        for npc in npcs:
            npc.update(dt)

        # Update player
        player.update(dt, throttle=accel, brake=brake, steer=steer, handbrake=handbrake)

        # clamp player into map bounds
        player.pos.x = clamp(player.pos.x, 0, MAP_W)
        player.pos.y = clamp(player.pos.y, 0, MAP_H)

        # mission logic update
        if missions:
            m = missions[current_mission]
            if not m.active and not m.completed:
                m.start(now)
            # special handling for timed mission: if failed it becomes inactive and player can retry/skip
            if m.type == 'timed':
                time_left = m.params.get('time_limit', 0) - (now - m.start_time)
                if time_left <= 0 and not m.completed:
                    # failed timed mission, mark inactive so player can retry/skip
                    m.active = False
            # tail mission needs dt updates to accumulate
            if m.type == 'tail':
                m.tail_update_dt(dt, player, npcs)
            else:
                m.update(now, player, npcs)
            # if mission completed, advance to next after short delay
            if m.completed:
                # mark mission as completed (keeps state) and auto-advance
                current_mission = (current_mission + 1) % len(missions)
                missions[current_mission].start(now)

        # camera follow
        camera.follow(player.pos)

        # Draw world portion to screen
        screen.fill((0,0,0))
        screen.blit(world, (-camera.pos.x, -camera.pos.y))

        # Draw mission markers / items / targets
        if missions:
            m = missions[current_mission]
            if m.type == 'reach' or m.type == 'timed':
                target = Vector2(m.params['point'])
                screen_pos = target - camera.pos
                pygame.draw.circle(screen, (255, 215, 0), (int(screen_pos.x), int(screen_pos.y)), m.params.get('radius', 36))
                pygame.draw.circle(screen, (255, 255, 255), (int(screen_pos.x), int(screen_pos.y)), m.params.get('radius', 36), 2)
            elif m.type == 'collect':
                for i,p in enumerate(m.params.get('items', [])):
                    collected = m.params.get('collected', [False]*len(m.params.get('items', [])))[i]
                    color = (120,120,120) if collected else (255,200,40)
                    screen_pos = Vector2(p) - camera.pos
                    pygame.draw.rect(screen, color, (screen_pos.x-10, screen_pos.y-10, 20, 20))
            elif m.type == 'tail':
                npc = m.params.get('npc')
                if npc is not None:
                    screen_pos = npc.car.pos - camera.pos
                    pygame.draw.circle(screen, (220,80,60), (int(screen_pos.x), int(screen_pos.y)), 8)

        # Draw NPCs and player
        for npc in npcs:
            npc.draw(screen, camera.pos)
        player.draw(screen, camera.pos)

        # HUD
        speed_text = font.render(f"Speed: {int(player.vel.length())} px/s", True, (255,255,255))
        screen.blit(speed_text, (12, 12))
        instr = font.render("WASD / Arrows = Drive • SPACE = handbrake • M = minimap • F5 save • F9 load • N next mission", True, (220,220,220))
        screen.blit(instr, (12, SCREEN_H - 28))

        # Mission HUD
        if missions:
            m = missions[current_mission]
            title = font.render(f"Mission: {m.title}", True, (255,255,255))
            screen.blit(title, (12, 40))
            # status details
            if m.type == 'reach':
                status = font.render("Objective: Reach the marker.", True, (200,200,200))
                screen.blit(status, (12, 64))
            elif m.type == 'collect':
                collected = sum(1 for c in m.params.get('collected', []) if c)
                total = len(m.params.get('items', []))
                status = font.render(f"Collected: {collected}/{total}", True, (200,200,200))
                screen.blit(status, (12, 64))
            elif m.type == 'timed':
                time_left = max(0.0, m.params.get('time_limit', 0) - (now - m.start_time))
                status = font.render(f"Time left: {int(time_left)}s", True, (200,200,200))
                screen.blit(status, (12, 64))
            elif m.type == 'tail':
                tail_t = m.params.get('tail_time', 0.0)
                req = m.params.get('required_time', 0.0)
                status = font.render(f"Tailing: {int(tail_t)}s / {int(req)}s", True, (200,200,200))
                screen.blit(status, (12, 64))

        # mission complete notification handled by progression earlier; show small text when last mission completed looped

        pygame.display.flip()


if __name__ == "__main__":
    main()