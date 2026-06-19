import sys
import time
sys.path.append(r"C:\Program Files\Epic Games\UE_5.8\Engine\Plugins\Experimental\PythonScriptPlugin\Content\Python")
from remote_execution import RemoteExecution, MODE_EXEC_FILE

script = """
import unreal
actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)

# Find the visualizer room to place the tree near it
# We know the Garden is at Y=150000. We will place it nearby.
willow = actor_subsystem.spawn_actor_from_class(unreal.StaticMeshActor.static_class(), unreal.Vector(1000, 151000, 100))
if willow:
    willow.set_actor_label("Placeholder_WeepingWillow")
    willow_comp = willow.get_component_by_class(unreal.StaticMeshComponent.static_class())
    
    # Try to load a tree mesh from Elderboom as a placeholder
    tree_mesh = unreal.EditorAssetLibrary.load_asset("/Game/ElderboomVillage/Meshes/Trees/SM_Tree_01")
    # Fallback to a basic shape if no tree found
    if not tree_mesh:
        tree_mesh = unreal.EditorAssetLibrary.load_asset("/Engine/BasicShapes/Cone")
        
    if tree_mesh and willow_comp:
        willow_comp.set_static_mesh(tree_mesh)
        # Scale it up to look majestic
        willow.set_actor_scale3d(unreal.Vector(5, 5, 8))
        
unreal.EditorLoadingAndSavingUtils.save_dirty_packages(False, True)
unreal.log("Placeholder Willow Tree Spawned!")
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
