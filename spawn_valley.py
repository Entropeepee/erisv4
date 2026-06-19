import sys
import time
sys.path.append(r"C:\Program Files\Epic Games\UE_5.8\Engine\Plugins\Experimental\PythonScriptPlugin\Content\Python")
from remote_execution import RemoteExecution, MODE_EXEC_FILE

script = """
import unreal

actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
level_subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)

# Check if already exists
all_actors = actor_subsystem.get_all_level_actors()
found = False
for actor in all_actors:
    if "ValleyOfTheAncient" in actor.get_actor_label() or isinstance(actor, unreal.LevelInstance):
        world_asset = actor.get_editor_property('world_asset')
        if world_asset and "AncientWorld" in world_asset.get_name():
            found = True
            break

if not found:
    world_asset = unreal.EditorAssetLibrary.load_asset('/Game/AncientContent/Maps/AncientWorld')
    if world_asset:
        actor = actor_subsystem.spawn_actor_from_class(unreal.LevelInstance.static_class(), unreal.Vector(150000, 0, 0))
        if actor:
            actor.set_actor_label("ValleyOfTheAncient_Instance")
            actor.set_editor_property('world_asset', world_asset)
            # We don't force load_level_instance() because it's 100GB, we let Unreal do it dynamically or the user can do it.
            unreal.log("Successfully spawned Valley of the Ancient as a Level Instance at X=150000.")
            
            # Move the Portal to face the Desert if we want, or create a second portal
            # Actually, let's leave it, the user can just fly there or walk.
            unreal.EditorLoadingAndSavingUtils.save_dirty_packages(True, True)
    else:
        unreal.log_error("Could not load /Game/AncientContent/Maps/AncientWorld! Ensure the editor has finished Discovering Assets.")
else:
    unreal.log("Valley of the Ancient is already spawned in this level!")
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
