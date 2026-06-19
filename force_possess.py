import sys
import time
sys.path.append(r"C:\Program Files\Epic Games\UE_5.8\Engine\Plugins\Experimental\PythonScriptPlugin\Content\Python")
from remote_execution import RemoteExecution, MODE_EXEC_FILE

remote_exec = RemoteExecution()
remote_exec.start()
time.sleep(1.0)

nodes = remote_exec.remote_nodes
if not nodes:
    sys.exit(1)

node_id = nodes[0]['node_id']
remote_exec.open_command_connection(node_id)

unreal_script = """
import unreal

subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
if subsystem:
    # Spawn a DefaultPawn
    location = unreal.Vector(0, 0, 100)
    rotation = unreal.Rotator(0, 0, 0)
    new_pawn = subsystem.spawn_actor_from_class(unreal.DefaultPawn.static_class(), location, rotation)
    
    if new_pawn:
        new_pawn.set_actor_label("Antigravity_Debug_Pawn")
        new_pawn.set_editor_property('auto_possess_player', unreal.AutoReceiveInput.PLAYER0)
        print("SUCCESS: Spawned a debug pawn and forced player possession to it.")
    else:
        print("ERROR: Failed to spawn pawn.")
"""

res = remote_exec.run_command(unreal_script, exec_mode=MODE_EXEC_FILE)
print("Execution result:", res)
remote_exec.stop()
