extends Node2D
## Pretty replay of dump_trajectory.py JSON.
## Keep Python as the brain; this scene only visualizes.

@export_file("*.json") var trajectory_path: String = "res://data/large_best.json"
@export var playback_speed: float = 1.0
@export var loop_episodes: bool = true
@export var auto_advance_episodes: bool = true
@export var trail_length: int = 96
@export var show_predictions: bool = true
@export var quit_when_done: bool = false
## Framing: overview (default) = fixed full arena; agent_goal / agent = follow crops.
@export var frame_mode: String = "overview"
## Minimum world span (width) so late-episode closeness doesn't collapse the crop.
@export var min_view_span: float = 280.0
## Extra world padding around the agent–goal box.
@export var frame_padding: float = 56.0
## Legacy half-span for --frame agent (ignored in agent_goal / overview).
@export var view_radius: float = 200.0
## Extra visual scale on disks (sim radii unchanged).
@export var sprite_scale: float = 1.25
## Skip tiny episodes so the reel doesn't feel like instant goal taps.
@export var min_episode_frames: int = 50
## Comma-separated outcomes to keep (empty = all). Showcase uses "success".
@export var outcome_filter: String = "success"

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
var _viewport_size: float = 1920.0


func _ready() -> void:
	_camera = $Camera2D
	_hud = $HUD/Panel/Label
	var vp := get_viewport().get_visible_rect().size
	_viewport_size = minf(vp.x, vp.y)
	# CLI: godot -- --trajectory path.json --frame agent_goal --outcomes success --speed 0.85
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
			"--view-radius":
				if i + 1 < args.size():
					view_radius = float(args[i + 1])
					frame_mode = "agent"
					i += 1
			"--min-view-span":
				if i + 1 < args.size():
					min_view_span = float(args[i + 1])
					i += 1
			"--min-frames":
				if i + 1 < args.size():
					min_episode_frames = int(args[i + 1])
					i += 1
			"--outcomes":
				if i + 1 < args.size():
					outcome_filter = str(args[i + 1])
					i += 1
			"--frame":
				if i + 1 < args.size():
					frame_mode = str(args[i + 1])
					i += 1
			"--overview":
				frame_mode = "overview"
			"--quit-when-done":
				quit_when_done = true
		i += 1
	if not _load_trajectory(trajectory_path):
		_hud.text = "Failed to load:\n%s" % trajectory_path
		push_error(_hud.text)
		return
	_apply_static_camera_if_needed()
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
	var raw_eps: Array = _data.get("episodes", [])
	var allowed: Dictionary = {}
	if outcome_filter.strip_edges() != "":
		for part in outcome_filter.split(","):
			allowed[part.strip_edges()] = true
	_episodes = []
	for ep in raw_eps:
		var frames: Array = ep.get("frames", [])
		var outcome := str(ep.get("outcome", ""))
		if not allowed.is_empty() and not allowed.has(outcome):
			continue
		if frames.size() >= min_episode_frames:
			_episodes.append(ep)
	# If everything was filtered out, keep the longest *allowed* (or any) episode.
	if _episodes.is_empty() and not raw_eps.is_empty():
		var best: Dictionary = {}
		var best_n := 0
		for ep2 in raw_eps:
			var outcome2 := str(ep2.get("outcome", ""))
			if not allowed.is_empty() and not allowed.has(outcome2):
				continue
			var n: int = ep2.get("frames", []).size()
			if n > best_n:
				best_n = n
				best = ep2
		if best.is_empty():
			# Fall back to longest overall so the viewer still opens.
			for ep3 in raw_eps:
				var n3: int = ep3.get("frames", []).size()
				if n3 > best_n:
					best_n = n3
					best = ep3
		if not best.is_empty():
			_episodes.append(best)
	var b: Dictionary = _meta.get("boundary", {})
	_boundary = Vector2(float(b.get("width", 400)), float(b.get("height", 400)))
	_fps = float(_meta.get("fps", 30))
	return _episodes.size() > 0


func _apply_static_camera_if_needed() -> void:
	if frame_mode == "overview":
		_camera.position_smoothing_enabled = false
		var margin := 1.08
		var zoom_fit: float = minf(
			_viewport_size / (_boundary.x * margin),
			_viewport_size / (_boundary.y * margin)
		)
		_camera.zoom = Vector2(zoom_fit, zoom_fit)
		_camera.position = _boundary * 0.5
	elif frame_mode == "agent":
		_camera.position_smoothing_enabled = true
		var margin := 1.08
		var span: float = maxf(view_radius * 2.0, min_view_span)
		var zoom_fit2: float = _viewport_size / (span * margin)
		_camera.zoom = Vector2(zoom_fit2, zoom_fit2)
		_camera.position = _boundary * 0.5
	else:
		_camera.position_smoothing_enabled = true


func _update_follow_camera(agent_pos: Vector2, goal_pos: Vector2) -> void:
	if frame_mode == "overview":
		# Stay locked on the full board — never chase the agent.
		_camera.position = _boundary * 0.5
		return
	var margin := 1.06
	var center: Vector2
	var span: float
	if frame_mode == "agent_goal":
		center = (agent_pos + goal_pos) * 0.5
		var dx: float = absf(agent_pos.x - goal_pos.x)
		var dy: float = absf(agent_pos.y - goal_pos.y)
		span = maxf(dx, dy) + frame_padding * 2.0
		span = clampf(span, min_view_span, maxf(_boundary.x, _boundary.y))
	else:
		# agent-only follow
		center = agent_pos
		span = maxf(view_radius * 2.0, min_view_span)
	var zoom_fit: float = _viewport_size / (span * margin)
	_camera.zoom = Vector2(zoom_fit, zoom_fit)
	# Keep the frame inside the arena when possible.
	var half: float = span * 0.5
	center.x = clampf(center.x, half, _boundary.x - half)
	center.y = clampf(center.y, half, _boundary.y - half)
	_camera.position = center


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
					await get_tree().create_timer(0.5).timeout
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
	var goal: Dictionary = fr.get("goal", {})
	var pos := Vector2(float(agent.get("x", 0)), float(agent.get("y", 0)))
	var gpos := Vector2(float(goal.get("x", 0)), float(goal.get("y", 0)))
	_update_follow_camera(pos, gpos)
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
	var cam_s := frame_mode
	if frame_mode == "overview":
		cam_s = "full board"
	elif frame_mode == "agent_goal":
		cam_s = "agent+goal ≥%.0f" % min_view_span
	elif frame_mode == "agent":
		cam_s = "agent ±%.0f" % view_radius
	_hud.text = (
		"%s  |  %s  |  ep %d/%d  |  %s\n"
		% [str(_meta.get("algo", "?")).to_upper(), str(_meta.get("scenario", "?")), _ep_idx + 1, _episodes.size(), cam_s]
		+ "outcome: %s   frame %d/%d%s\n"
		% [str(ep.get("outcome", "?")), fi + 1, frames.size(), pause_s]
		+ "Space pause · N next · --overview (default)"
	)


func _draw() -> void:
	# Soft arena backdrop
	var pad := 24.0
	draw_rect(Rect2(Vector2(-pad, -pad), _boundary + Vector2(pad * 2, pad * 2)), Color(0.09, 0.12, 0.18))
	draw_rect(Rect2(Vector2.ZERO, _boundary), Color(0.12, 0.16, 0.24))
	# Grid — thin hairlines that stay readable when zoomed in
	var step := 40.0
	var grid_c := Color(0.18, 0.24, 0.34, 0.55)
	var grid_w := 1.25
	var x := 0.0
	while x <= _boundary.x:
		draw_line(Vector2(x, 0), Vector2(x, _boundary.y), grid_c, grid_w, true)
		x += step
	var y := 0.0
	while y <= _boundary.y:
		draw_line(Vector2(0, y), Vector2(_boundary.x, y), grid_c, grid_w, true)
		y += step
	draw_rect(Rect2(Vector2.ZERO, _boundary), Color(0.45, 0.7, 1.0, 0.55), false, 3.0)

	if _episodes.is_empty():
		return
	var frames: Array = _episodes[_ep_idx].get("frames", [])
	if frames.is_empty():
		return
	var fr: Dictionary = frames[clampi(int(_frame_idx), 0, frames.size() - 1)]
	var agent_r := float(_meta.get("agent_radius", 8)) * sprite_scale
	var obs_r := float(_meta.get("obstacle_radius", 8)) * sprite_scale
	var goal_r := float(_meta.get("goal_radius", 6)) * sprite_scale

	# Agent trail
	if _agent_trail.size() >= 2:
		for i in range(1, _agent_trail.size()):
			var a := float(i) / float(_agent_trail.size())
			draw_line(
				_agent_trail[i - 1],
				_agent_trail[i],
				Color(0.55, 0.85, 1.0, a * 0.9),
				3.5,
				true
			)

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
			draw_line(from, to, Color(0.85, 0.9, 1.0, 0.4), 2.0, true)
			draw_circle(to, 5.0, Color(1.0, 0.55, 0.4) if noisy else Color(0.4, 0.95, 0.6))

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
	draw_arc(gpos, goal_r * 2.2, 0, TAU, 96, Color(0.35, 0.95, 0.55, 0.5), 2.5, true)

	# Agent
	var agent: Dictionary = fr.get("agent", {})
	var apos := Vector2(float(agent.get("x", 0)), float(agent.get("y", 0)))
	_draw_soft_disk(apos, agent_r, Color(0.95, 0.95, 0.98))
	# Action wedge
	var action := int(agent.get("action", 0))
	var ang := deg_to_rad(float(action * 45))
	var tip := apos + Vector2(cos(ang), sin(ang)) * (agent_r + 18.0)
	draw_line(apos, tip, Color(1.0, 0.85, 0.25, 0.95), 3.5, true)
	draw_circle(tip, 4.0, Color(1.0, 0.85, 0.25))


func _draw_soft_disk(pos: Vector2, radius: float, color: Color) -> void:
	draw_circle(pos, radius * 1.65, Color(color.r, color.g, color.b, 0.16))
	draw_circle(pos, radius, color)
	draw_arc(pos, radius, 0, TAU, 96, Color(1, 1, 1, 0.4), 2.0, true)


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
	var steps: int = 32
	for s in range(steps + 1):
		var t: float = float(s) / float(steps)
		var a: float = angle - spread + (2.0 * spread * t)
		points.append(from + Vector2(cos(a), sin(a)) * radius)
	draw_colored_polygon(points, color)
