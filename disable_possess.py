import sys
import time
sys.path.append(r"C:\Program Files\Epic Games\UE_5.8\Engine\Plugins\Experimental\PythonScriptPlugin\Content\Python")
from remote_execution import RemoteExecution, MODE_EXEC_FILE

remote_exec = RemoteExecution()
remote_exec.start()
time.sleep(1.0)

nodes = remote_exec.remote_nodes
if not nodes:
    print("Could not find any Unreal Engine instances.")
    remote_exec.stop()
    sys.exit(1)

node_id = nodes[0]['node_id']
remote_exec.open_command_connection(node_id)

unreal_script = """
import unreal

subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
if subsystem:
    actors = subsystem.get_all_level_actors()
    fixed_count = 0
    for actor in actors:
        try:
            val = actor.get_editor_property('auto_possess_player')
            if val != unreal.AutoReceiveInput.DISABLED:
                print(f"Found actor with AutoPossess enabled: {actor.get_name()} (Class: {actor.get_class().get_name()})")
                actor.set_editor_property('auto_possess_player', unreal.AutoReceiveInput.DISABLED)
                fixed_count += 1
        except Exception as e:
            pass
    print(f"SUCCESS: Disabled AutoPossess on {fixed_count} instances.")
else:
    print("ERROR: Could not get EditorActorSubsystem.")
"""

res = remote_exec.run_command(unreal_script, exec_mode=MODE_EXEC_FILE)
print("Execution result:", res)
remote_exec.stop()
