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
    
    # Spawn Elderboom
    level_inst = actor_subsystem.spawn_actor_from_class(unreal.LevelInstance.static_class(), unreal.Vector(5000, 5000, 0), unreal.Rotator(0,0,0))
    if level_inst:
        world_asset = unreal.EditorAssetLibrary.load_asset('/Game/ElderboomVillage/Maps/L_ElderboomVillage')
        if world_asset:
            try:
                level_inst.set_editor_property('world_asset', world_asset)
            except Exception as e:
                # sometimes it requires a SoftObjectPath
                level_inst.set_editor_property('world_asset', unreal.SoftObjectPath('/Game/ElderboomVillage/Maps/L_ElderboomVillage.L_ElderboomVillage'))
            level_inst.set_actor_label("Elderboom_Sublevel")
            unreal.log("Spawned Elderboom level instance.")
    
    # Spawn Eris
    bp_natalia_asset = unreal.EditorAssetLibrary.load_blueprint_class('/Game/MetaHumans/Natalia/BP_Natalia')
    if bp_natalia_asset:
        eris = actor_subsystem.spawn_actor_from_class(bp_natalia_asset, unreal.Vector(500, 0, 100), unreal.Rotator(0, 0, 0))
        if eris:
            eris.set_actor_label("Eris_AI")
            eris.set_editor_property('auto_possess_player', unreal.AutoReceiveInput.DISABLED)
            eris.set_editor_property('auto_possess_ai', unreal.AutoPossessAI.PLACED_IN_WORLD_OR_SPAWNED)
            unreal.log("Spawned Eris.")
            
    unreal.EditorLevelLibrary.save_current_level()
    print("SUCCESS: Stitched scenes and spawned Eris.")
except Exception as e:
    print(f"ERROR: {e}")
"""

res = remote_exec.run_command(unreal_script, exec_mode=MODE_EXEC_FILE)
print("Execution result:", res)
remote_exec.stop()
