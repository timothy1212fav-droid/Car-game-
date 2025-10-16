import random
import json
import time
import pygame
import sys

# Initialize Pygame for 4K high-resolution interface
pygame.init()
SCREEN_WIDTH, SCREEN_HEIGHT = 3840, 2160  # 4K resolution
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
pygame.display.set_caption("FC25 Open World Soccer (4K Render) — With Leagues")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 72)  # Scaled for 4K

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Game state
player_pos = [400, 300]
player_health = 100
budget = 50000
season = 15
score = [0, 0]
match_time = 0
match_playing = False
in_vehicle = False
current_vehicle = None
nearest_vehicle = None
nearest_phone = None
player_has_phone = False
current_call = None
career = {"active": False, "player": None, "club": "FC 25"}
leagues = {}
nations = {}
club_players = []
npcs = []
police_units = []
vehicles = {"cars": [], "planes": [], "helis": []}
world_places = {}
phones = []
pedestrians = []

# List of all countries (approximately 195)
countries = [
    "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Antigua and Barbuda", "Argentina", "Armenia", "Australia", "Austria",
    "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin", "Bhutan",
    "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei", "Bulgaria", "Burkina Faso", "Burundi", "Cabo Verde", "Cambodia",
    "Cameroon", "Canada", "Central African Republic", "Chad", "Chile", "China", "Colombia", "Comoros", "Congo", "Costa Rica",
    "Croatia", "Cuba", "Cyprus", "Czech Republic", "Denmark", "Djibouti", "Dominica", "Dominican Republic", "Ecuador", "Egypt",
    "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia", "Eswatini", "Ethiopia", "Fiji", "Finland", "France", "Gabon",
    "Gambia", "Georgia", "Germany", "Ghana", "Greece", "Grenada", "Guatemala", "Guinea", "Guinea-Bissau", "Guyana",
    "Haiti", "Honduras", "Hungary", "Iceland", "India", "Indonesia", "Iran", "Iraq", "Ireland", "Israel",
    "Italy", "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kiribati", "Kuwait", "Kyrgyzstan", "Laos",
    "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya", "Liechtenstein", "Lithuania", "Luxembourg", "Madagascar", "Malawi",
    "Malaysia", "Maldives", "Mali", "Malta", "Marshall Islands", "Mauritania", "Mauritius", "Mexico", "Micronesia", "Moldova",
    "Monaco", "Mongolia", "Montenegro", "Morocco", "Mozambique", "Myanmar", "Namibia", "Nauru", "Nepal", "Netherlands",
    "New Zealand", "Nicaragua", "Niger", "Nigeria", "North Korea", "North Macedonia", "Norway", "Oman", "Pakistan", "Palau",
    "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Poland", "Portugal", "Qatar", "Romania", "Russia",
    "Rwanda", "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Samoa", "San Marino", "Sao Tome and Principe", "Saudi Arabia", "Senegal", "Serbia",
    "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Slovenia", "Solomon Islands", "Somalia", "South Africa", "South Korea", "South Sudan",
    "Spain", "Sri Lanka", "Sudan", "Suriname", "Sweden", "Switzerland", "Syria", "Taiwan", "Tajikistan", "Tanzania",
    "Thailand", "Timor-Leste", "Togo", "Tonga", "Trinidad and Tobago", "Tunisia", "Turkey", "Turkmenistan", "Tuvalu", "Uganda",
    "Ukraine", "United Arab Emirates", "United Kingdom", "United States", "Uruguay", "Uzbekistan", "Vanuatu", "Vatican City", "Venezuela", "Vietnam",
    "Yemen", "Zambia", "Zimbabwe"
]

def generate_travel_list():
    travel_list = random.choices(countries, k=200)
    print("Here is a list of 200 countries to travel to:")
    for i, country in enumerate(travel_list, start=1):
        print(f"{i}. {country}")

# Classes
class Player:
    def __init__(self, name, age, pos, rating, xp=0, seasons=0, goals=0, appearances=0):
        self.name = name
        self.age = age
        self.pos = pos
        self.rating = rating
        self.xp = xp
        self.seasons = seasons
        self.goals = goals
        self.appearances = appearances

class Team:
    def __init__(self, name):
        self.name = name
        self.players = []

class League:
    def __init__(self, name):
        self.name = name
        self.teams = []

class Referee:
    def __init__(self, name, strictness, corruption):
        self.name = name
        self.strictness = strictness
        self.corruption = corruption

class Match:
    def __init__(self, home_team, away_team, referee):
        self.home_team = home_team
        self.away_team = away_team
        self.referee = referee
        self.elapsed = 0
        self.duration = 90
        self.home = 0
        self.away = 0
        self.home_fouls = 0
        self.away_fouls = 0
        self.home_yellows = 0
        self.away_yellows = 0
        self.home_reds = 0
        self.away_reds = 0
        self.home_goals_effect = 0
        self.away_goals_effect = 0
        self.bribed = False
        self.bribe_amount = 0
        self.bribe_success = False
        self.bribe_caught = False

class Vehicle:
    def __init__(self, type_, x, y):
        self.type = type_
        self.x = x
        self.y = y
        self.speed = 0
        self.fuel = 100 if type_ == "car" else 200
        self.for_sale = False
        self.price = 0
        self.owner = "none"

class NPC:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = random.uniform(-0.5, 0.5)
        self.vy = random.uniform(-0.5, 0.5)
        self.alive = True

class Police:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.state = "patrol"
        self.target_x = x + random.uniform(-20, 20)
        self.target_y = y + random.uniform(-20, 20)

# Functions
def create_random_player(name_prefix='Player', idx=0):
    return Player(
        name=f"{name_prefix} {random.randint(1, 9000)}",
        age=16 + random.randint(0, 20),
        pos=random.choice(['GK', 'DEF', 'MID', 'FWD']),
        rating=40 + random.randint(0, 40)
    )

def setup_leagues():
    global leagues
    for i in range(6):
        league_name = f"League {chr(65 + i)}"
        leagues[league_name] = League(league_name)
        for j in range(15):
            team_name = f"{league_name} - Club {j + 1}"
            team = Team(team_name)
            leagues[league_name].teams.append(team)

def setup_nations():
    global nations
    for i in range(1, 121):
        nations[f"Nation {i}"] = {"id": i, "name": f"Nation {i}"}

def setup_players():
    global club_players
    for i in range(200):
        club_players.append(create_random_player('Player', i))

def setup_npcs(count=50):
    global npcs, pedestrians
    for _ in range(count):
        x = random.uniform(-100, 100)
        y = random.uniform(-100, 100)
        npc = NPC(x, y)
        npcs.append(npc)
        pedestrians.append(npc)

def setup_police(count=6):
    global police_units
    for _ in range(count):
        x = random.uniform(-90, 90)
        y = random.uniform(-90, 90)
        police_units.append(Police(x, y))

def setup_vehicles():
    global vehicles
    # Cars
    for _ in range(18):
        x = random.uniform(-110, 110)
        y = random.uniform(-110, 110)
        vehicles["cars"].append(Vehicle("car", x, y))
    # Planes
    for i in range(4):
        x = 40 + (i - 1.5) * 6
        y = -10 - 6 - i * 2
        plane = Vehicle("plane", x, y)
        plane.for_sale = True
        plane.price = 50000 + random.randint(0, 150000)
        plane.owner = "dealership"
        vehicles["planes"].append(plane)
    # Helis
    for i in range(6):
        x = 40 + (i - 3) * 3
        y = -10 + 8 + random.uniform(-0.5, 0.5)
        vehicles["helis"].append(Vehicle("heli", x, y))

def setup_world_places():
    global world_places
    # Houses
    world_places["houses"] = []
    base_x = -80
    for i in range(12):
        x = base_x + (i % 6) * 7
        y = -30 + (i // 6) * 8
        world_places["houses"].append({"x": x, "y": y, "owner": "player" if i == 0 else f"npc_{i}", "type": "house"})
    # Hospital
    world_places["hospital"] = {"x": 60, "y": -40, "type": "hospital"}
    # Prison
    world_places["prison"] = {"x": 70, "y": 40, "type": "prison"}
    # Airstrip
    world_places["airstrip"] = {"x": 40, "y": -10, "type": "airstrip"}

def setup_phones(count=14):
    global phones
    for _ in range(count):
        x = random.uniform(-150, 150)
        y = random.uniform(-150, 150)
        phones.append({"x": x, "y": y, "picked": False})

def spawn_phones():
    setup_phones()

def check_proximity():
    global nearest_vehicle, nearest_phone
    nearest_vehicle = None
    min_dist = float('inf')
    for veh_list in vehicles.values():
        for veh in veh_list:
            dist = ((veh.x - player_pos[0]/10) ** 2 + (veh.y - player_pos[1]/10) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                nearest_vehicle = veh
    if min_dist > 3:
        nearest_vehicle = None

    nearest_phone = None
    min_dist = float('inf')
    for phone in phones:
        if not phone["picked"]:
            dist = ((phone["x"] - player_pos[0]/10) ** 2 + (phone["y"] - player_pos[1]/10) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                nearest_phone = phone
    if min_dist > 3:
        nearest_phone = None

def pick_up_phone():
    global player_has_phone
    if player_has_phone:
        return
    if nearest_phone:
        nearest_phone["picked"] = True
        player_has_phone = True

def drop_phone():
    global player_has_phone
    if not player_has_phone:
        return
    phone = {"x": player_pos[0]/10 + random.uniform(-1, 1), "y": player_pos[1]/10 + random.uniform(-1, 1), "picked": False}
    phones.append(phone)
    player_has_phone = False

def call_contact(name, type_):
    global current_call
    if not player_has_phone or current_call:
        return
    current_call = {"name": name, "start": time.time()}
    print(f"Calling {name}...")

def hang_up():
    global current_call
    if current_call:
        print(f"Call with {current_call['name']} ended.")
        current_call = None

def update_npcs(dt):
    for npc in npcs:
        if not npc.alive:
            continue
        npc.x += npc.vx * dt
        npc.y += npc.vy * dt
        if abs(npc.x) > 220:
            npc.vx *= -1
        if abs(npc.y) > 220:
            npc.vy *= -1

def update_police(dt):
    for pol in police_units:
        if pol.state == "patrol":
            dir_x = pol.target_x - pol.x
            dir_y = pol.target_y - pol.y
            dist = (dir_x**2 + dir_y**2)**0.5
            if dist < 1:
                pol.target_x = pol.x + random.uniform(-40, 40)
                pol.target_y = pol.y + random.uniform(-40, 40)
            else:
                pol.x += (dir_x / dist) * 5 * dt
                pol.y += (dir_y / dist) * 5 * dt
        elif pol.state == "chase":
            dir_x = player_pos[0]/10 - pol.x
            dir_y = player_pos[1]/10 - pol.y
            dist = (dir_x**2 + dir_y**2)**0.5
            if dist < 2:
                player_pos[0] = world_places["prison"]["x"] * 10
                player_pos[1] = world_places["prison"]["y"] * 10
                pol.state = "patrol"
            else:
                pol.x += (dir_x / dist) * 8 * dt
                pol.y += (dir_y / dist) * 8 * dt

def generate_foul_event():
    committer = random.choice(["home", "away"])
    severity = random.random()
    in_box = random.random() < 0.12
    return {"committer": committer, "severity": severity, "in_box": in_box, "time": int(match_time)}

def handle_foul_event(foul, match):
    ref = match.referee
    card_chance = foul["severity"] * (0.35 + ref.strictness * 0.65)
    if match.bribed and match.bribe_success:
        if foul["committer"] == "home":
            card_chance *= 0.4
        else:
            card_chance *= 1.3
    match.__dict__[f"{foul['committer']}_fouls"] += 1

    if foul["in_box"]:
        if random.random() < min(0.9, 0.45 + foul["severity"] * 0.5 + (ref.strictness - 0.5) * 0.2):
            winner = "away" if foul["committer"] == "home" else "home"
            print(f"Penalty awarded to {winner} side!")
            score_prob = 0.75
            if match.bribed and match.bribe_success and winner == "home":
                score_prob += 0.15
            if random.random() < score_prob:
                match.__dict__[winner] += 1
                print("Penalty scored!")
            else:
                print("Penalty missed!")
            return

    if random.random() < 0.06 + foul["severity"] * 0.06:
        winner = "away" if foul["committer"] == "home" else "home"
        fk_prob = 0.18 + foul["severity"] * 0.18
        if match.bribed and match.bribe_success and winner == "home":
            fk_prob += 0.12
        if random.random() < fk_prob:
            match.__dict__[winner] += 1
            print(f"Free kick converted by {winner} side!")

    if random.random() < card_chance:
        red_threshold = 0.88
        if foul["severity"] > red_threshold or random.random() < foul["severity"] * 0.25:
            match.__dict__[f"{foul['committer']}_reds"] += 1
            print(f"{foul['committer']} player sent off (RED)!")
            match.__dict__[f"{foul['committer']}_goals_effect"] += 0.25
        else:
            match.__dict__[f"{foul['committer']}_yellows"] += 1
            print(f"{foul['committer']} player given a YELLOW.")
            if match.__dict__[f"{foul['committer']}_yellows"] >= 2:
                match.__dict__[f"{foul['committer']}_reds"] += 1
                print(f"{foul['committer']} player received second yellow -> RED!")
                match.__dict__[f"{foul['committer']}_goals_effect"] += 0.25

def play_next_fixture():
    global match_playing, match_time, score
    if match_playing:
        return
    ref = Referee("M. Dupont", 0.7, 0.05)  # Simplified, pick one
    bribe_input = input(f"Referee assigned: {ref.name} (strictness {int(ref.strictness*100)}%). Enter bribe amount (0 for none): ")
    bribe_amt = max(0, int(float(bribe_input or 0)))
    global budget
    bribe_success = False
    bribe_caught = False
    if bribe_amt > 0:
        if bribe_amt > budget:
            print("Insufficient budget for that bribe.")
            return
        budget -= bribe_amt
        bribe_success = random.random() < min(1, ref.corruption + bribe_amt / 50000)
        if random.random() < 0.06 + bribe_amt / 100000:
            bribe_caught = True
            fine = min(bribe_amt * 2, budget)
            budget = max(0, budget - fine)
            print("Bribe was detected! Fine applied.")
        else:
            print("Bribe accepted." if bribe_success else "Bribe failed.")
    match = Match("Your Team", "Opponent", ref)
    match.bribed = bribe_amt > 0
    match.bribe_amount = bribe_amt
    match.bribe_success = bribe_success
    match.bribe_caught = bribe_caught
    match_playing = True
    match_time = 0
    score = [0, 0]
    print("Match started!")

def end_current_match():
    global match_playing, season, budget
    if not match_playing:
        return
    match_playing = False
    res = score[0] - score[1]
    delta = 0
    if res > 0:
        delta = 10000
        print("You won! Budget increased.")
    elif res == 0:
        delta = 2000
        print("Draw. Small income.")
    else:
        delta = -5000
        print("You lost. Budget decreased.")
    budget += delta
    season += 1
    print(f"Final score: {score[0]} - {score[1]}")

def update_match(dt):
    global match_time, score
    if not match_playing:
        return
    match_time += dt
    if int(match_time) % 1 == 0 and random.random() < 0.12:
        foul = generate_foul_event()
        handle_foul_event(foul, Match("home", "away", Referee("", 0.5, 0.1)))  # Simplified
    if random.random() < 0.04:
        side = random.choice(["home", "away"])
        score[0 if side == "home" else 1] += 1
        print(f"Goal! {side} side scored.")
    if match_time >= 90:
        end_current_match()

def create_career_player():
    name = input("Enter your player name: ")
    if not name:
        return
    pl = create_random_player(name, 0)
    pl.rating = 50 + random.randint(0, 30)
    career["player"] = pl.__dict__
    career["active"] = True
    print(f"Career created for {pl.name}.")

def retire_career_player():
    if not career["player"]:
        return
    print(f"{career['player']['name']} retired.")
    career["player"] = None
    career["active"] = False

def save_game():
    state = {
        "player_pos": player_pos,
        "player_health": player_health,
        "budget": budget,
        "season": season,
        "career": career,
        "player_has_phone": player_has_phone,
        "phones": phones,
        "vehicles": {k: [v.__dict__ for v in lst] for k, lst in vehicles.items()},
        "npcs": [n.__dict__ for n in npcs],
        "police_units": [p.__dict__ for p in police_units]
    }
    with open("fc25_save.json", "w") as f:
        json.dump(state, f)
    print("Game saved.")

def load_game():
    global player_pos, player_health, budget, season, career, player_has_phone, phones, vehicles, npcs, police_units
    try:
        with open("fc25_save.json", "r") as f:
            state = json.load(f)
        player_pos = state.get("player_pos", player_pos)
        player_health = state.get("player_health", player_health)
        budget = state.get("budget", budget)
        season = state.get("season", season)
        career = state.get("career", career)
        player_has_phone = state.get("player_has_phone", player_has_phone)
        phones = state.get("phones", phones)
        vehicles = {k: [Vehicle(**v) for v in lst] for k, lst in state.get("vehicles", {}).items()}
        npcs = [NPC(**n) for n in state.get("npcs", [])]
        police_units = [Police(**p) for p in state.get("police_units", [])]
        print("Game loaded.")
    except FileNotFoundError:
        print("No save file found.")

def draw():
    screen.fill(BLACK)
    # Draw player (scaled for 4K)
    pygame.draw.circle(screen, GREEN, (int(player_pos[0] * 4.8), int(player_pos[1] * 3.6)), 40)
    # Draw NPCs (scaled for 4K)
    for npc in npcs:
        pygame.draw.circle(screen, WHITE, (int(npc.x * 48 + 1920), int(npc.y * 36 + 1080)), 20)
    # Draw police (scaled for 4K)
    for pol in police_units:
        pygame.draw.circle(screen, BLUE, (int(pol.x * 48 + 1920), int(pol.y * 36 + 1080)), 20)
    # Draw vehicles (scaled for 4K)
    for veh_list in vehicles.values():
        for veh in veh_list:
            color = RED if veh.type == "car" else YELLOW
            pygame.draw.rect(screen, color, (int(veh.x * 48 + 1920), int(veh.y * 36 + 1080), 40, 40))
    # Draw phones (scaled for 4K)
    for phone in phones:
        if not phone["picked"]:
            pygame.draw.circle(screen, BLUE, (int(phone["x"] * 48 + 1920), int(phone["y"] * 36 + 1080)), 12)
    # UI (scaled for 4K)
    ui_text = f"Health: {player_health} Budget: ${budget} Season: {season} Score: {score[0]} - {score[1]} Time: {int(match_time)}"
    screen.blit(font.render(ui_text, True, WHITE), (40, 40))
    if player_has_phone:
        screen.blit(font.render("Phone: Ready", True, WHITE), (40, 160))
    if nearest_vehicle:
        screen.blit(font.render("Press E to enter vehicle", True, WHITE), (40, 280))
    if nearest_phone:
        screen.blit(font.render("Press P to pick up phone", True, WHITE), (40, 400))
    pygame.display.flip()

def main():
    setup_leagues()
    setup_nations()
    setup_players()
    setup_npcs()
    setup_police()
    setup_vehicles()
    setup_world_places()
    spawn_phones()

    running = True
    last_time = time.time()
    while running:
        dt = time.time() - last_time
        last_time = time.time()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    player_pos[1] -= 20  # Scaled for 4K
                elif event.key == pygame.K_s:
                    player_pos[1] += 20  # Scaled for 4K
                elif event.key == pygame.K_a:
                    player_pos[0] -= 20  # Scaled for 4K
                elif event.key == pygame.K_d:
                    player_pos[0] += 20  # Scaled for 4K
                elif event.key == pygame.K_p:
                    if player_has_phone:
                        if current_call:
                            hang_up()
                        else:
                            # Simplified call
                            pass
                    else:
                        pick_up_phone()
                elif event.key == pygame.K_e:
                    if nearest_vehicle:
                        in_vehicle = not in_vehicle
                        current_vehicle = nearest_vehicle if in_vehicle else None
                elif event.key == pygame.K_h:
                    # Simplified house interaction
                    pass
                elif event.key == pygame.K_m:
                    play_next_fixture()
                elif event.key == pygame.K_n:
                    end_current_match()
                elif event.key == pygame.K_c:
                    create_career_player()
                elif event.key == pygame.K_r:
                    retire_career_player()
                elif event.key == pygame.K_s:
                    save_game()
                elif event.key == pygame.K_l:
                    load_game()

        check_proximity()
        update_npcs(dt)
        update_police(dt)
        update_match(dt)
        if current_call and time.time() - current_call["start"] > 7:
            hang_up()
        draw()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
