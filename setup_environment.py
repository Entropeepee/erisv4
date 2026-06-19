import sys
import time
sys.path.append(r"C:\Program Files\Epic Games\UE_5.8\Engine\Plugins\Experimental\PythonScriptPlugin\Content\Python")
from remote_execution import RemoteExecution, MODE_EXEC_FILE

script = """
import unreal

actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)

# 1. Day/Night Cycle (Directional Light + Sky Atmosphere)
dir_light = None
sky_atm = None

for actor in actor_subsystem.get_all_level_actors():
    if isinstance(actor, unreal.DirectionalLight):
        dir_light = actor
    elif isinstance(actor, unreal.SkyAtmosphere):
        sky_atm = actor

if not dir_light:
    dir_light = actor_subsystem.spawn_actor_from_class(unreal.DirectionalLight.static_class(), unreal.Vector(0,0,1000))
    dir_light.set_actor_label("SunLight")

if dir_light:
    # Make it movable
    dir_light.light_component.set_editor_property("mobility", unreal.ComponentMobility.MOVABLE)
    dir_light.light_component.set_editor_property("intensity", 10.0)
    dir_light.light_component.set_editor_property("atmosphere_sun_light", True)

if not sky_atm:
    sky_atm = actor_subsystem.spawn_actor_from_class(unreal.SkyAtmosphere.static_class(), unreal.Vector(0,0,0))
    sky_atm.set_actor_label("Atmosphere")

# Spawn simple rain audio actor
audio_actor = actor_subsystem.spawn_actor_from_class(unreal.AmbientSound.static_class(), unreal.Vector(-1000, -1000, 500))
audio_actor.set_actor_label("Ambient_RainWind")
# Assign a default sound if possible, but we don't know the exact path of a rain sound. We will just leave it empty for user to drop a sound into it, or use a starter content sound.
sound_asset = unreal.EditorAssetLibrary.load_asset('/Engine/VREditor/Sounds/UI/Laser_Hover_Cue') # Placeholder
if sound_asset:
    audio_actor.audio_component.set_editor_property("sound", sound_asset)

unreal.log("Environment Setup Complete: Added Movable Sun, Atmosphere, and Audio Actor.")
"""

remote_exec = RemoteExecution()
remote_exec.start()
time.sleep(1.0)
nodes = remote_exec.remote_nodes
if nodes:
    node_id = nodes[0]['node_id']
    remote_exec.open_command_connection(node_id)
    res = remote_exec.run_command(script, exec_mode=MODE_EXEC_FILE)
    print(res)
    remote_exec.stop()
else:
    print("No nodes found.")
