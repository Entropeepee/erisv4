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

try:
    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    
    world_asset = unreal.EditorAssetLibrary.load_asset('/Game/Maps/RomanCave')
    if not world_asset:
        unreal.log_error("Could not load RomanCave map.")
    else:
        new_loc = unreal.Vector(-25000, 15000, -50)
        level_inst = actor_subsystem.spawn_actor_from_class(unreal.LevelInstance.static_class(), new_loc)
        
        if level_inst:
            level_inst.set_actor_label("RomanCave_Biome")
            level_inst.set_world_asset(world_asset)
            unreal.log("Successfully spawned Roman Cave biome.")
            unreal.EditorLevelLibrary.save_current_level()
except Exception as e:
    unreal.log_error(f"Error: {e}")
"""

res = remote_exec.run_command(unreal_script, exec_mode=MODE_EXEC_FILE)
print("Execution result:", res)
remote_exec.stop()
