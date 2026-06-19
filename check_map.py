import sys
import time
sys.path.append(r"C:\Program Files\Epic Games\UE_5.8\Engine\Plugins\Experimental\PythonScriptPlugin\Content\Python")
from remote_execution import RemoteExecution, MODE_EXEC_FILE

script = """
import unreal

actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
level_subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)

current_world = level_subsystem.get_current_level()
world_name = current_world.get_outer().get_name() if current_world else "Unknown"

all_actors = actor_subsystem.get_all_level_actors()
actor_names = [actor.get_actor_label() for actor in all_actors]

sphere_found = any("VegasSphere" in name for name in actor_names)
ancient_found = any("Ancient" in name for name in actor_names)
roman_found = any("Roman" in name for name in actor_names)

unreal.log(f"Current Map: {world_name}")
unreal.log(f"Total Actors: {len(actor_names)}")
unreal.log(f"Vegas Sphere Found: {sphere_found}")
unreal.log(f"Roman Cave Found: {roman_found}")
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
