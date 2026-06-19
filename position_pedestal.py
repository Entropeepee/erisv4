import sys
import time
sys.path.append(r"C:\Program Files\Epic Games\UE_5.8\Engine\Plugins\Experimental\PythonScriptPlugin\Content\Python")
from remote_execution import RemoteExecution, MODE_EXEC_FILE

script = """
import unreal

actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
all_actors = actor_subsystem.get_all_level_actors()

found_widget = False
for actor in all_actors:
    # Usually it's named something like 'WBP_Pedestal_C'
    if "WBP_Pedestal" in actor.get_actor_label() or "WBP_Pedestal" in actor.get_name():
        actor.set_actor_label("Console_Pedestal_Actor")
        actor.set_actor_location(unreal.Vector(-10000, 10000, 1050), False, True)
        actor.set_actor_rotation(unreal.Rotator(0, 90, 0), False)
        actor.set_actor_scale3d(unreal.Vector(2, 2, 2))
        found_widget = True
        unreal.log("Found and repositioned the Console Pedestal.")
        break

if not found_widget:
    unreal.log_warning("Could not find the Pedestal Widget in the level. Make sure it was dragged in.")
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
    print("Could not connect to Unreal Engine.")
