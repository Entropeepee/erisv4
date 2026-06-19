import sys
import time
sys.path.append(r"C:\Program Files\Epic Games\UE_5.8\Engine\Plugins\Experimental\PythonScriptPlugin\Content\Python")
from remote_execution import RemoteExecution, MODE_EXEC_FILE

script = """
import unreal

actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)

all_actors = actor_subsystem.get_all_level_actors()
for actor in all_actors:
    name = actor.get_actor_label()
    if "Biome" in name or "TeleportTarget" in name or "Fire_Teleporter" in name:
        unreal.log(f"Found stitched actor: {name} at {actor.get_actor_location()}")
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
