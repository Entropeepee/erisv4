import sys
import time
sys.path.append(r"C:\Program Files\Epic Games\UE_5.8\Engine\Plugins\Experimental\PythonScriptPlugin\Content\Python")
from remote_execution import RemoteExecution, MODE_EXEC_FILE

script = """
import unreal

actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)

# Helper to avoid duplicates
def spawn_if_not_exists(actor_class, location, label):
    for a in actor_subsystem.get_all_level_actors():
        if a.get_actor_label() == label:
            return a
    actor = actor_subsystem.spawn_actor_from_class(actor_class, location)
    if actor:
        actor.set_actor_label(label)
    return actor

# 1. Stitch Valley of the Ancient (The Desert!)
ancient_asset = unreal.EditorAssetLibrary.load_asset('/Game/AncientContent/Maps/AncientWorld')
if ancient_asset:
    ancient_inst = spawn_if_not_exists(unreal.LevelInstance.static_class(), unreal.Vector(150000, 0, 0), "Biome_ValleyOfTheAncient")
    if ancient_inst:
        ancient_inst.set_editor_property('world_asset', ancient_asset)
        unreal.log("Stitched Valley of the Ancient at X=150000")

# 2. Stitch Dream World Garden
garden_asset = unreal.EditorAssetLibrary.load_asset('/Game/DreamWorld_Garden')
if garden_asset:
    garden_inst = spawn_if_not_exists(unreal.LevelInstance.static_class(), unreal.Vector(0, 150000, 0), "Biome_DreamWorldGarden")
    if garden_inst:
        garden_inst.set_editor_property('world_asset', garden_asset)
        unreal.log("Stitched Dream World Garden at Y=150000")

# 3. Spawn a Teleport Trigger Box & Targets
garden_target = spawn_if_not_exists(unreal.TargetPoint.static_class(), unreal.Vector(0, 150000, 500), "TeleportTarget_Garden")
ancient_target = spawn_if_not_exists(unreal.TargetPoint.static_class(), unreal.Vector(150000, 0, 500), "TeleportTarget_Ancient")
fire_trigger = spawn_if_not_exists(unreal.TriggerBox.static_class(), unreal.Vector(0, 0, 100), "Fire_Teleporter_Trigger")
if fire_trigger:
    fire_trigger.set_actor_scale3d(unreal.Vector(2, 2, 2))

unreal.EditorLoadingAndSavingUtils.save_dirty_packages(True, True)
unreal.log("Biome Stitching Complete! Teleport targets and trigger spawned.")
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
