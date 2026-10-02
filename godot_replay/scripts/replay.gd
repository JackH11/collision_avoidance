extends Node2D
## Pretty replay of dump_trajectory.py JSON.
## Keep Python as the brain; this scene only visualizes.

@export_file("*.json") var trajectory_path: String = "res://data/large_best.json"
@export var playback_speed: float = 1.0
@export var loop_episodes: bool = true
@export var auto_advance_episodes: bool = true
@export var trail_length: int = 48
@export var show_predictions: bool = true
@export var quit_when_done: bool = false

var _data: Dictionary = {}
var _episodes: Array = []
var _ep_idx: int = 0
var _frame_idx: float = 0.0
var _paused: bool = false
var _agent_trail: PackedVector2Array = PackedVector2Array()
var _meta: Dictionary = {}
var _boundary: Vector2 = Vector2(400, 400)
var _fps: float = 30.0
var _camera: Camera2D
var _hud: Label
var _done_quitting: bool = false


func _ready() -> void:
	_camera = $Camera2D
	_hud = $HUD/Panel/Label
	# CLI override: godot -- --trajectory path.json --quit-when-done
	var args := OS.get_cmdline_user_args()
	var i := 0
	while i < args.size():
		match args[i]:
			"--trajectory":
				if i + 1 < args.size():
					trajectory_path = args[i + 1]
					i += 1
			"--speed":
				if i + 1 < args.size():
					playback_speed = float(args[i + 1])
					i += 1
			"--quit-when-done":
				quit_when_done = true
		i += 1
	if not _load_trajectory(trajectory_path):
		_hud.text = "Failed to load:\n%s" % trajectory_path
		push_error(_hud.text)
		return
	_reset_episode(0)


func _load_trajectory(path: String) -> bool:
	var abs_path := path
	if path.begins_with("res://") or path.begins_with("user://"):
		abs_path = ProjectSettings.globalize_path(path)
	if not FileAccess.file_exists(path) and not FileAccess.file_exists(abs_path):
		# Fall back to absolute filesystem path for dumps outside the project.
		if not FileAccess.file_exists(path):
			return false
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		f = FileAccess.open(abs_path, FileAccess.READ)
	if f == null:
		return false
	var text := f.get_as_text()
	var parsed: Variant = JSON.parse_string(text)
	if typeof(parsed) != TYPE_DICTIONARY:
		return false
	_data = parsed
	_meta = _data.get("meta", {})
	_episodes = _data.get("episodes", [])
	var b: Dictionary = _meta.get("boundary", {})
	_boundary = Vector2(float(b.get("width", 400)), float(b.get("height", 400)))
	_fps = float(_meta.get("fps", 30))
	# Fit camera so the arena fills most of the 960px viewport
	var margin := 1.15
	var zoom_fit: float = minf(960.0 / (_boundary.x * margin), 960.0 / (_boundary.y * margin))
	_camera.zoom = Vector2(zoom_fit, zoom_fit)
	_camera.position = _boundary * 0.5
	return _episodes.size() > 0


func _reset_episode(idx: int) -> void:
	_ep_idx = clampi(idx, 0, maxi(_episodes.size() - 1, 0))
	_frame_idx = 0.0
	_agent_trail = PackedVector2Array()
	_update_hud()
	queue_redraw()


func _process(delta: float) -> void:
	if _episodes.is_empty() or _paused:
		return
	var ep: Dictionary = _episodes[_ep_idx]
	var frames: Array = ep.get("frames", [])
	if frames.is_empty():
		return
	_frame_idx += delta * _fps * playback_speed
	if _frame_idx >= frames.size() - 1:
		_frame_idx = float(frames.size() - 1)
		if auto_advance_episodes:
			var next := _ep_idx + 1
			if next >= _episodes.size():
				if quit_when_done and not _done_quitting:
					_done_quitting = true
					# Give MovieWriter a moment to flush last frames
					await get_tree().create_timer(0.35).timeout
					get_tree().quit()
					return
				if loop_episodes:
					next = 0
				else:
					_paused = true
					return
			_reset_episode(next)
			return
	var fi := int(_frame_idx)
	var fr: Dictionary = frames[fi]
	var agent: Dictionary = fr.get("agent", {})
	var pos := Vector2(float(agent.get("x", 0)), float(agent.get("y", 0)))
	_camera.position = pos
	_push_trail(pos)
	_update_hud()
	queue_redraw()


func _push_trail(pos: Vector2) -> void:
	_agent_trail.append(pos)
	while _agent_trail.size() > trail_length:
		_agent_trail.remove_at(0)


func _unhandled_input(event: InputEvent) -> void:
	if event.is_action_pressed("toggle_pause"):
		_paused = not _paused
		_update_hud()
	elif event.is_action_pressed("next_episode"):
		_reset_episode((_ep_idx + 1) % max(_episodes.size(), 1))


func _update_hud() -> void:
	if _hud == null or _episodes.is_empty():
		return
	var ep: Dictionary = _episodes[_ep_idx]
	var frames: Array = ep.get("frames", [])
	var fi := clampi(int(_frame_idx), 0, max(frames.size() - 1, 0))
	var pause_s := "  PAUSED" if _paused else ""
	_hud.text = (
		"%s  |  %s  |  ep %d/%d\n"
		% [str(_meta.get("algo", "?")).to_upper(), str(_meta.get("scenario", "?")), _ep_idx + 1, _episodes.size()]
		+ "outcome: %s   frame %d/%d%s\n"
		% [str(ep.get("outcome", "?")), fi + 1, frames.size(), pause_s]
		+ "Space pause · N next episode · dump: dump_trajectory.py"
	)


func _draw() -> void:
	# Soft arena backdrop
	var pad := 24.0
	draw_rect(Rect2(Vector2(-pad, -pad), _boundary + Vector2(pad * 2, pad * 2)), Color(0.09, 0.12, 0.18))
	draw_rect(Rect2(Vector2.ZERO, _boundary), Color(0.12, 0.16, 0.24))
	# Grid
	var step := 40.0
	var grid_c := Color(0.18, 0.24, 0.34, 0.7)
	var x := 0.0
	while x <= _boundary.x:
		draw_line(Vector2(x, 0), Vector2(x, _boundary.y), grid_c, 1.0)
		x += step
	var y := 0.0
	while y <= _boundary.y:
		draw_line(Vector2(0, y), Vector2(_boundary.x, y), grid_c, 1.0)
		y += step
	draw_rect(Rect2(Vector2.ZERO, _boundary), Color(0.45, 0.7, 1.0, 0.55), false, 2.5)

	if _episodes.is_empty():
		return
	var frames: Array = _episodes[_ep_idx].get("frames", [])
	if frames.is_empty():
		return
	var fr: Dictionary = frames[clampi(int(_frame_idx), 0, frames.size() - 1)]
	var agent_r := float(_meta.get("agent_radius", 8))
	var obs_r := float(_meta.get("obstacle_radius", 8))
	var goal_r := float(_meta.get("goal_radius", 6))

	# Agent trail
	if _agent_trail.size() >= 2:
		for i in range(1, _agent_trail.size()):
			var a := float(i) / float(_agent_trail.size())
			draw_line(_agent_trail[i - 1], _agent_trail[i], Color(0.55, 0.85, 1.0, a * 0.85), 3.0)

	# Predictions / uncertainty wedges
	if show_predictions:
		var preds: Array = fr.get("predictions", [])
		var obstacles: Array = fr.get("obstacles", [])
		for i in range(mini(preds.size(), obstacles.size())):
			var p: Dictionary = preds[i]
			var o: Dictionary = obstacles[i]
			var from := Vector2(float(o.get("x", 0)), float(o.get("y", 0)))
			var to := Vector2(float(p.get("x", 0)), float(p.get("y", 0)))
			var noisy := bool(o.get("noise", false))
			var cone_c := Color(1.0, 0.45, 0.35, 0.22) if noisy else Color(0.35, 0.85, 0.55, 0.22)
			_draw_uncertainty_cone(from, to, float(p.get("std_x", 1)) + float(p.get("std_y", 1)), cone_c)
			draw_line(from, to, Color(0.85, 0.9, 1.0, 0.35), 1.5)
			draw_circle(to, 4.0, Color(1.0, 0.55, 0.4) if noisy else Color(0.4, 0.95, 0.6))

	# Obstacles
	for o in fr.get("obstacles", []):
		var od: Dictionary = o
		var pos := Vector2(float(od.get("x", 0)), float(od.get("y", 0)))
		var noisy := bool(od.get("noise", false))
		var fill := Color(0.95, 0.35, 0.35) if noisy else Color(0.35, 0.55, 0.95)
		_draw_soft_disk(pos, obs_r, fill)

	# Goal
	var goal: Dictionary = fr.get("goal", {})
	var gpos := Vector2(float(goal.get("x", 0)), float(goal.get("y", 0)))
	_draw_soft_disk(gpos, goal_r * 1.35, Color(0.35, 0.95, 0.55))
	draw_arc(gpos, goal_r * 2.2, 0, TAU, 48, Color(0.35, 0.95, 0.55, 0.45), 2.0)

	# Agent
	var agent: Dictionary = fr.get("agent", {})
	var apos := Vector2(float(agent.get("x", 0)), float(agent.get("y", 0)))
	_draw_soft_disk(apos, agent_r, Color(0.95, 0.95, 0.98))
	# Action wedge
	var action := int(agent.get("action", 0))
	var ang := deg_to_rad(float(action * 45))
	var tip := apos + Vector2(cos(ang), sin(ang)) * (agent_r + 16.0)
	draw_line(apos, tip, Color(1.0, 0.85, 0.25, 0.95), 3.0)
	draw_circle(tip, 3.0, Color(1.0, 0.85, 0.25))


func _draw_soft_disk(pos: Vector2, radius: float, color: Color) -> void:
	draw_circle(pos, radius * 1.55, Color(color.r, color.g, color.b, 0.18))
	draw_circle(pos, radius, color)
	draw_arc(pos, radius, 0, TAU, 40, Color(1, 1, 1, 0.35), 1.5)


func _draw_uncertainty_cone(from: Vector2, to: Vector2, variance: float, color: Color) -> void:
	var delta: Vector2 = to - from
	var radius: float = delta.length()
	if radius < 1.0:
		return
	var angle: float = delta.angle()
	var denom: float = 1.0 + maxf(variance, 0.0)
	var spread: float = (PI * 0.5) / denom
	var points: PackedVector2Array = PackedVector2Array()
	points.append(from)
	var steps: int = 16
	for s in range(steps + 1):
		var t: float = float(s) / float(steps)
		var a: float = angle - spread + (2.0 * spread * t)
		points.append(from + Vector2(cos(a), sin(a)) * radius)
	draw_colored_polygon(points, color)
