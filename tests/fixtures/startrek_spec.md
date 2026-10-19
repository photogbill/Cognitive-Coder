# Begin 4: A Deterministic Tactical Starship Simulation

Build a complete, playable 2D simultaneous-turn ("WeGo") tactical starship combat simulation in Python 3.11 using pygame and pytest. The simulation models vector kinematics, rotational inertia, 6-sector hexagonal shields, warp engine heat, command-line order parsing, and utility-based AI captains across four factions (Federation, Klingon, Romulan, and Orion).

## 1. Hard Requirements

1. Combat, movement, shield absorption, and subsystem damage MUST be 100% deterministic with zero use of random number generators anywhere in the simulation core.
2. You MUST NEVER import or call the `random` module in `src/models.py`, `src/physics.py`, `src/parser.py`, or `src/ai.py`.
3. Every combat turn MUST resolve simultaneously across exactly 10 sequential physics sub-ticks (`SUBTICKS_PER_TURN = 10`, with `dt = 0.1` per sub-tick).
4. Ship headings MUST use navigational bearings in degrees where `0.0` is North (`0, -1`), `90.0` is East (`1, 0`), `180.0` is South (`0, 1`), and `270.0` is West (`-1, 0`), normalized to `[0.0, 360.0)`.
5. Relative helm commands MUST accept signed degree offsets from `-360.0` to `+360.0` and absolute course commands MUST set the target heading in `[0.0, 360.0)`.
6. Each starship MUST maintain exactly 6 hexagonal shield facings (`len(shields) == 6`) indexed `0` through `5` in 60-degree relative bearing sectors around the hull.
7. Damage exceeding an impacted shield sector's current energy MUST bleed through to hull integrity and deterministically degrade internal subsystems in fixed priority order.
8. Pushing warp speed above a ship's cruising warp rating MUST accumulate engine heat each sub-tick, and reaching `100.0` heat MUST force target warp to `0.0` and damage the warp drive.
9. You MUST NEVER call `input()` or block on standard input in any module or test.
10. `src/models.py`, `src/physics.py`, `src/parser.py`, and `src/ai.py` MUST NOT import `pygame` so the simulation core and test suite run completely headless.
11. Every public function and dataclass field pinned in the architecture skeleton MUST keep its exact name, parameter order, and return type.
12. All unit tests in `tests/test_physics.py`, `tests/test_parser.py`, and `tests/test_ai.py` MUST assert exact deterministic outputs derived from the module functions.

## 2. Project File Manifest

The build plan comprises 7 source files and 3 test files:

* `src/__init__.py` — Package marker exporting version metadata.
* `src/models.py` — Shared dataclasses, enums/constants, and factory functions for the 4 factions and ship classes.
* `src/physics.py` — Pure deterministic 10-sub-tick kinematics, bearing/sector geometry, ballistic torpedo movement, beam attenuation, shield impact resolution, and subsystem degradation.
* `src/parser.py` — Lexical command-line parser converting player text commands and abbreviations into structured `CommandOrder` instances.
* `src/ai.py` — Deterministic AI captain decision engine weighted by faction personality profiles (`Bravery`, `Aggression`, `Fanaticism`, `Loyalty`).
* `src/render.py` — Pygame 640x480+ split-screen tactical vector HUD renderer (fleet list, central radar with 6-sided shield rings and velocity vectors, right-side thermal/weapon bars, bottom CLI prompt).
* `src/main.py` — Application entry point and Pygame event loop supporting `--headless-demo` for automated verification.
* `tests/test_physics.py` — Unit tests for kinematics, 6-facing shield sector mapping, weapon arcs, warp heat, and deterministic damage bleed-through.
* `tests/test_parser.py` — Unit tests for full and abbreviated command syntax, multi-command chaining, and invalid command handling.
* `tests/test_ai.py` — Unit tests for AI helm selection, shield rotation, and weapon firing under different personality weights.

## 3. Shared Data Contracts (`src/models.py`)

Define the following data structures and constants in `src/models.py`:

* Constants:
  * `SUBTICKS_PER_TURN: int = 10`
  * `SUBTICK_DT: float = 0.1`
  * `WORLD_RADIUS: float = 10000.0`
  * `FACTIONS: tuple[str, ...] = ("Federation", "Klingon", "Romulan", "Orion")`
* `Personality`: dataclass with `bravery: float`, `aggression: float`, `fanaticism: float`, `loyalty: float` (each in `0.0` to `1.0`).
* `WeaponBank`: dataclass with:
  * `bank_id: int`
  * `kind: str` (`"phaser"`, `"disruptor"`, `"photon"`, `"plasma"`)
  * `arc_center_deg: float` (relative to ship bow, e.g., `0.0` forward, `180.0` aft)
  * `arc_width_deg: float` (total firing cone width in degrees, e.g., `120.0` or `240.0`)
  * `max_charge: float`
  * `charge: float`
  * `range_limit: float`
  * `locked_target_id: str` (empty string `""` when unlocked)
  * `integrity: float = 100.0` (`0.0` to `100.0`)
* `Torpedo`: dataclass with:
  * `torpedo_id: int`
  * `owner_id: str`
  * `target_id: str`
  * `kind: str` (`"photon"` or `"plasma"`)
  * `x: float`
  * `y: float`
  * `heading_deg: float`
  * `speed: float`
  * `damage_yield: float`
  * `ttl_subticks: int`
* `Ship`: dataclass with:
  * `ship_id: str` (short tactical code such as `"En"`, `"Kl"`, `"Ro"`, `"Or"`)
  * `name: str`
  * `faction: str`
  * `ship_class: str`
  * `x: float`
  * `y: float`
  * `heading_deg: float`
  * `target_heading_deg: float`
  * `warp: float`
  * `target_warp: float`
  * `cruising_warp: float`
  * `max_warp: float`
  * `mass: float`
  * `turn_rate_deg_per_sec: float`
  * `shields: list[float]` (6 floats: `[0]` Forward `330..30`, `[1]` Starboard-Bow `30..90`, `[2]` Starboard-Quarter `90..150`, `[3]` Aft `150..210`, `[4]` Port-Quarter `210..270`, `[5]` Port-Bow `270..330`)
  * `max_shield_per_sector: float`
  * `hull: float`
  * `max_hull: float`
  * `warp_heat: float = 0.0`
  * `Reserve_energy: float = 100.0`
  * `cloaked: bool = False`
  * `can_cloak: bool = False`
  * `subsystems: dict[str, float]` with keys `"warp"`, `"phasers"`, `"torpedoes"`, `"sensors"`, `"shields"`, each initialized to `100.0`
  * `weapons: list[WeaponBank]`
  * `personality: Personality`
  * `fleet_order: str = "attack"`
* `CommandOrder`: dataclass with:
  * `ship_id: str`
  * `action: str` (`"course"`, `"helm"`, `"warp"`, `"lock"`, `"unlock"`, `"fire"`, `"shield_transfer"`, `"cloak"`, `"decloak"`, `"scan"`, `"fleet"`, `"status"`, `"noop"`)
  * `value: float = 0.0`
  * `target_id: str = ""`
  * `bank_ids: tuple[int, ...] = ()`
  * `sector: int = 0`
  * `message: str = ""`
* `WorldState`: dataclass with:
  * `turn: int = 1`
  * `ships: dict[str, Ship]`
  * `torpedoes: list[Torpedo]`
  * `combat_log: list[str]`
  * `next_torpedo_id: int = 1`
* Factory functions in `src/models.py`:
  * `create_ship(ship_id: str, name: str, faction: str, x: float, y: float, heading_deg: float) -> Ship`
  * `create_default_skirmish() -> WorldState` (spawns Federation flagship `"En"` *Enterprise* at `(0.0, 400.0)` heading `0.0` and Klingon cruiser `"Kl"` *Klothos* at `(0.0, -400.0)` heading `180.0`).

## 4. Deterministic Physics & Combat Engine (`src/physics.py`)

Implement pure functions in `src/physics.py`:

* `normalize_angle(deg: float) -> float`: Maps any angle to `[0.0, 360.0)`.
* `shortest_angle_diff(current_deg: float, target_deg: float) -> float`: Signed delta in `(-180.0, 180.0]`.
* `bearing_between(x1: float, y1: float, x2: float, y2: float) -> float`: Navigational bearing in `[0.0, 360.0)` from `(x1, y1)` to `(x2, y2)` using `math.atan2(x2 - x1, -(y2 - y1))`.
* `distance_between(x1: float, y1: float, x2: float, y2: float) -> float`: Euclidean distance `math.hypot(x2 - x1, y2 - y1)`.
* `impact_shield_sector(target_heading_deg: float, shooter_x: float, shooter_y: float, target_x: float, target_y: float) -> int`:
  * Compute incoming bearing from target to shooter: `inc = bearing_between(target_x, target_y, shooter_x, shooter_y)`.
  * Compute relative angle off target's bow: `rel = normalize_angle(inc - target_heading_deg)`.
  * Map `rel` into sector `0..5`: `int(((rel + 30.0) % 360.0) // 60.0)`.
* `in_weapon_arc(ship_heading_deg: float, bank: WeaponBank, target_bearing_deg: float) -> bool`:
  * Compute absolute arc center `normalize_angle(ship_heading_deg + bank.arc_center_deg)`.
  * Return `abs(shortest_angle_diff(arc_center, target_bearing_deg)) <= (bank.arc_width_deg / 2.0)`.
* `apply_damage(target: Ship, sector: int, damage: float, log: list[str]) -> None`:
  * Clamp `sector` to `0..5`. Absorbed energy is `min(target.shields[sector], max(0.0, damage))`.
  * Subtract absorbed energy from `target.shields[sector]`.
  * Overflow `overflow = max(0.0, damage - absorbed)` subtracts directly from `target.hull`.
  * If `overflow > 0.0`, degrade subsystems deterministically in order `("shields", "weapons_or_phasers" -> "phasers", "torpedoes", "warp", "sensors")` by `overflow * 0.25` each, clamped to `0.0`.
* `apply_orders(world: WorldState, orders: list[CommandOrder]) -> None`:
  * Execute `"course"`, `"helm"`, `"warp"`, `"lock"`, `"unlock"`, `"shield_transfer"`, `"cloak"`, `"decloak"`, `"scan"`, and `"fire"`.
  * Cloaked ships cannot fire weapons while `cloaked` is `True`.
  * Firing a beam weapon (`"phaser"` or `"disruptor"`) checks `bank.charge >= 50.0`, range `<= bank.range_limit`, and `in_weapon_arc`. Beam damage is `bank.charge * (1.0 - 0.5 * (dist / bank.range_limit)) * (ship.subsystems["phasers"] / 100.0)`, applied immediately to `impact_shield_sector`, resetting `bank.charge = 0.0`.
  * Firing a `"photon"` or `"plasma"` bank spawns a `Torpedo` at the ship's position aimed along the bearing to `locked_target_id` (if in arc and `bank.charge >= 80.0`), resetting `bank.charge = 0.0`.
* `step_subtick(world: WorldState, dt: float = SUBTICK_DT) -> None`:
  * Accelerate/decelerate `ship.warp` toward `ship.target_warp` (clamped by `max_warp * (ship.subsystems["warp"] / 100.0)`).
  * Effective turn rate is `ship.turn_rate_deg_per_sec / max(1.0, ship.warp * 0.5)` (ships at `warp <= 1.0` turn at full rate; high warp widens turn arcs). Rotate `heading_deg` toward `target_heading_deg`.
  * Advance ship coordinates by `speed = ship.warp * 25.0`: `x += math.sin(rad) * speed * dt`, `y -= math.cos(rad) * speed * dt`.
  * Update `warp_heat`: if `ship.warp > ship.cruising_warp`, add `(ship.warp - ship.cruising_warp) * 8.0 * dt`; otherwise cool by `4.0 * dt` down to `0.0`. If `warp_heat >= 100.0`, set `target_warp = 0.0`, `warp = 0.0`, and reduce `subsystems["warp"]` by `15.0`.
  * Regenerate each shield sector by `0.25 * dt * (ship.subsystems["shields"] / 100.0)` up to `max_shield_per_sector`, and recharge weapon banks by `5.0 * dt` up to `max_charge`.
  * Advance active torpedoes; if a torpedo comes within `35.0` units of its target (and target is alive), apply `torpedo.damage_yield` to `impact_shield_sector` and remove the torpedo; decrement `ttl_subticks` and expire at `0`.
* `resolve_turn(world: WorldState, player_orders: list[CommandOrder], ai_orders: list[CommandOrder]) -> WorldState`:
  * Apply `player_orders` and `ai_orders`, run `step_subtick` across `SUBTICKS_PER_TURN` iterations, increment `world.turn`, and return `world`.

## 5. Command Parser (`src/parser.py`)

Implement `parse_command_line(ship_id: str, line: str) -> list[CommandOrder]`:

* Split `line` on semicolons `;` so players can chain commands in a single turn (e.g., `he -45; wa 6; lo 1-2 Kl; fi 1-2`).
* Support case-insensitive full commands and 2-letter abbreviations:
  * `course <deg>` or `co <deg>` -> `action="course"`, `value=float(deg) % 360.0`
  * `helm <delta>` or `he <delta>` -> `action="helm"`, `value=max(-360.0, min(360.0, float(delta)))`
  * `warp <speed>` or `wa <speed>` -> `action="warp"`, `value=max(0.0, float(speed))`
  * `lock <banks> <target>` or `lo <banks> <target>` -> `action="lock"`, `bank_ids=parse_bank_spec(banks)`, `target_id=target`
  * `fire <banks>` or `fi <banks>` -> `action="fire"`, `bank_ids=parse_bank_spec(banks)`
  * `shield <sector> <amount>` or `sh <sector> <amount>` -> `action="shield_transfer"`, `sector=int(sector)`, `value=float(amount)`
  * `cloak` / `cl` and `decloak` / `de` -> `action="cloak"` / `"decloak"`
  * `scan <target>` or `sc <target>` -> `action="scan"`, `target_id=target`
* Implement `parse_bank_spec(spec: str) -> tuple[int, ...]` supporting single ints (`"1"` -> `(1,)`), ranges (`"1-3"` -> `(1, 2, 3)`), and comma lists (`"1,3"` -> `(1, 3)`).
* Unknown or malformed tokens return a `CommandOrder` with `action="noop"` and a descriptive `message` string without raising unhandled exceptions.

## 6. Deterministic AI Captain Engine (`src/ai.py`)

Implement `compute_ai_orders(world: WorldState, ai_ship_id: str) -> list[CommandOrder]`:

* Select the nearest enemy ship (`faction != ai_ship.faction` and `hull > 0.0`). If none exists, return `[CommandOrder(ship_id=ai_ship_id, action="warp", value=0.0)]`.
* Compute range and bearing to target.
* **Shield rotation & bravery check:** Identify the AI ship's facing sector currently pointed at the enemy. If that shield sector is below `25.0` and `ai_ship.personality.fanaticism < 0.8`, angle the helm by `+60.0` degrees to present a stronger adjacent shield sector. Otherwise, if `distance > 350.0` and `ai_ship.personality.bravery >= 0.5`, set course toward the enemy intercept point; if `hull < 30.0` and `ai_ship.personality.fanaticism < 0.5`, set course away from the enemy (`(bearing + 180.0) % 360.0`).
* **Warp heat management:** Choose target warp equal to `ai_ship.cruising_warp` when `warp_heat > 60.0`, or `min(ai_ship.max_warp, ai_ship.cruising_warp + ai_ship.personality.aggression * 2.0)` when closing distance.
* **Weapons control:** Lock all charged weapons (`charge >= 50.0`) onto the target and issue a `"fire"` order for all banks whose firing arc covers the target bearing and are within `bank.range_limit`.

## 7. Tactical Vector Viewport (`src/render.py`) and Entry Point (`src/main.py`)

* In `src/render.py`, implement `TacticalRenderer`:
  * Render a classic 4-panel VGA tactical layout on a `pygame.Surface` (default `1024x768`):
    1. **Left Panel:** Fleet roster with ship IDs, faction colors, speed, heading, and hull percentage.
    2. **Center Panel:** Square 2D tactical radar centered on the player flagship (`"En"`), drawing range rings, ship directional chevrons, 6-segment hexagonal shield rings colored by sector strength (blue/green/amber/red), and torpedo vectors.
    3. **Right Panel:** Flagship subsystem integrity bars, 6-facing shield numeric readouts, weapon bank charge/lock gauges, and the `Warp Heat` thermal bar.
    4. **Bottom Panel:** Scrolling combat log and live `Order > ` command-line text prompt.
* In `src/main.py`, implement `main(argv: list[str] | None = None) -> int`:
  * Support `--headless-demo`: when passed, run 3 deterministic turns of `create_default_skirmish()` with scripted orders (`he 0; wa 4; lo 1-2 Kl; fi 1-2`), verify state progression, print a summary line, and return `0` without opening a display window.
  * In normal mode, initialize Pygame, open the tactical window, collect keystrokes into the command prompt, execute `resolve_turn` when `Enter` is pressed, and exit cleanly with `0` on `ESC` or `pygame.QUIT`.