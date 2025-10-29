"""
Automated Blender rendering script for DiffSHEG realtime wrapper output.
Runs Blender in headless mode - no GUI required.

Usage:
    blender -b assets/beat_visualize.blend --python render_realtime_output.py
    
Or with custom paths:
    blender -b assets/beat_visualize.blend --python render_realtime_output.py -- \
        --bvh results/realtime_test/bvh/realtime_output.bvh \
        --json results/realtime_test/face_json/realtime_output.json \
        --audio results/realtime_test/2_scott_0_3_3.wav \
        --output results/realtime_test/output_video.mp4
"""

import bpy
import sys
import os
import json
import mathutils

# Get script arguments (after the -- separator)
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []

# Default paths (adjust these or pass via command line)
DEFAULT_BVH = "results/realtime_test/bvh/realtime_output.bvh"
DEFAULT_JSON = "results/realtime_test/face_json/realtime_output.json"
DEFAULT_AUDIO = "results/realtime_test/2_scott_0_3_3.wav"
DEFAULT_OUTPUT = "results/realtime_test/output_video.mp4"

# Parse command line arguments
bvh_path = DEFAULT_BVH
json_path = DEFAULT_JSON
audio_path = DEFAULT_AUDIO
output_path = DEFAULT_OUTPUT

i = 0
while i < len(argv):
    if argv[i] == "--bvh" and i + 1 < len(argv):
        bvh_path = argv[i + 1]
        i += 2
    elif argv[i] == "--json" and i + 1 < len(argv):
        json_path = argv[i + 1]
        i += 2
    elif argv[i] == "--audio" and i + 1 < len(argv):
        audio_path = argv[i + 1]
        i += 2
    elif argv[i] == "--output" and i + 1 < len(argv):
        output_path = argv[i + 1]
        i += 2
    else:
        i += 1

print("="*60)
print("DiffSHEG Realtime Output Renderer")
print("="*60)
print(f"BVH path:    {bvh_path}")
print(f"JSON path:   {json_path}")
print(f"Audio path:  {audio_path}")
print(f"Output path: {output_path}")
print("="*60)

# Verify files exist
if not os.path.exists(bvh_path):
    print(f"ERROR: BVH file not found: {bvh_path}")
    sys.exit(1)
if not os.path.exists(json_path):
    print(f"ERROR: JSON file not found: {json_path}")
    sys.exit(1)
if not os.path.exists(audio_path):
    print(f"ERROR: Audio file not found: {audio_path}")
    sys.exit(1)

# Clear existing objects (keep only camera and lights if needed)
print("\nClearing scene...")
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import BVH file
print(f"\nImporting BVH: {bvh_path}")
bpy.ops.import_anim.bvh(filepath=bvh_path, use_fps_scale=True)

# Get the armature created by BVH import
armature = None
for obj in bpy.context.selected_objects:
    if obj.type == 'ARMATURE':
        armature = obj
        break

if armature is None:
    print("ERROR: No armature found after BVH import")
    sys.exit(1)

print(f"Found armature: {armature.name}")

# Load facial expression JSON
print(f"\nLoading facial JSON: {json_path}")
with open(json_path, 'r') as f:
    facial_data = json.load(f)

print(f"Loaded {len(facial_data['frames'])} facial expression frames")

# Apply facial expressions as shape keys (if avatar has them)
# This part depends on your avatar setup - you may need to adapt this
# For now, we'll just log the data
print(f"Facial blendshape names: {', '.join(facial_data['names'][:5])}... ({len(facial_data['names'])} total)")

# Add a simple mesh for visualization (you can replace with actual avatar mesh)
# If you have the BEAT avatar mesh, load it here instead
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.2, location=(0, 0, 1.7))
head = bpy.context.active_object
head.name = "Head"

# Set up camera
bpy.ops.object.camera_add(location=(0, -3, 1.5), rotation=(1.5708, 0, 0))
camera = bpy.context.active_object
camera.data.lens = 50
bpy.context.scene.camera = camera

# Set up lighting
bpy.ops.object.light_add(type='SUN', location=(5, 5, 5))
sun = bpy.context.active_object
sun.data.energy = 2.0

# Add audio to scene
print(f"\nAdding audio: {audio_path}")
if not bpy.context.scene.sequence_editor:
    bpy.context.scene.sequence_editor_create()

sound = bpy.context.scene.sequence_editor.sequences.new_sound(
    name="Audio",
    filepath=audio_path,
    channel=1,
    frame_start=1
)

# Get animation frame range from BVH
if armature.animation_data and armature.animation_data.action:
    action = armature.animation_data.action
    frame_start = int(action.frame_range[0])
    frame_end = int(action.frame_range[1])
else:
    # Fallback: use facial data frame count
    frame_start = 1
    frame_end = len(facial_data['frames'])

print(f"\nAnimation frames: {frame_start} to {frame_end}")

# Set scene frame range
bpy.context.scene.frame_start = frame_start
bpy.context.scene.frame_end = frame_end
bpy.context.scene.frame_current = frame_start

# Set render settings
print("\nConfiguring render settings...")
# Try different render engines based on Blender version
render_engine_set = False
for engine in ['BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE', 'BLENDER_WORKBENCH', 'CYCLES']:
    try:
        bpy.context.scene.render.engine = engine
        print(f"Using render engine: {engine}")
        render_engine_set = True
        break
    except TypeError:
        continue

if not render_engine_set:
    print("Warning: Could not set render engine, using default")

bpy.context.scene.render.resolution_x = 1280
bpy.context.scene.render.resolution_y = 720
bpy.context.scene.render.fps = 15  # Match BEAT dataset FPS

# Set output format
bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
bpy.context.scene.render.ffmpeg.format = 'MPEG4'
bpy.context.scene.render.ffmpeg.codec = 'H264'
bpy.context.scene.render.ffmpeg.constant_rate_factor = 'HIGH'
bpy.context.scene.render.ffmpeg.ffmpeg_preset = 'GOOD'

# Audio settings
bpy.context.scene.render.ffmpeg.audio_codec = 'AAC'
bpy.context.scene.render.ffmpeg.audio_bitrate = 192

# Output path
output_dir = os.path.dirname(output_path)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

bpy.context.scene.render.filepath = output_path

print(f"\nOutput settings:")
print(f"  Resolution: {bpy.context.scene.render.resolution_x}x{bpy.context.scene.render.resolution_y}")
print(f"  FPS: {bpy.context.scene.render.fps}")
print(f"  Frames: {frame_start}-{frame_end} ({frame_end - frame_start + 1} frames)")
print(f"  Duration: {(frame_end - frame_start + 1) / bpy.context.scene.render.fps:.2f}s")

# Render animation
print("\n" + "="*60)
print("Starting render...")
print("="*60)
bpy.ops.render.render(animation=True, write_still=True)

print("\n" + "="*60)
print("Rendering complete!")
print(f"Video saved to: {os.path.abspath(output_path)}")
print("="*60)
