import sys
import time
sys.path.append(r"C:\Program Files\Epic Games\UE_5.8\Engine\Plugins\Experimental\PythonScriptPlugin\Content\Python")
from remote_execution import RemoteExecution, MODE_EXEC_FILE

script = """
import unreal

actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)

# 1. Spawn Vegas Sphere Outer
sphere_actor = actor_subsystem.spawn_actor_from_class(unreal.StaticMeshActor.static_class(), unreal.Vector(0,0,100000))
if sphere_actor:
    sphere_actor.set_actor_label("VegasSphere_Outer")
    mesh_comp = sphere_actor.static_mesh_component
    mesh = unreal.EditorAssetLibrary.load_asset('/Engine/BasicShapes/Sphere')
    mesh_comp.set_static_mesh(mesh)
    sphere_actor.set_actor_scale3d(unreal.Vector(500, 500, 500))

# 2. Spawn Platform Inside Sphere
platform_actor = actor_subsystem.spawn_actor_from_class(unreal.StaticMeshActor.static_class(), unreal.Vector(0,0,90000))
if platform_actor:
    platform_actor.set_actor_label("VegasSphere_Platform")
    mesh_comp = platform_actor.static_mesh_component
    mesh = unreal.EditorAssetLibrary.load_asset('/Engine/BasicShapes/Cylinder')
    mesh_comp.set_static_mesh(mesh)
    platform_actor.set_actor_scale3d(unreal.Vector(200, 200, 1))

# 3. Spawn Portal in the Garden (or Elderboom village on ground)
portal_actor = actor_subsystem.spawn_actor_from_class(unreal.StaticMeshActor.static_class(), unreal.Vector(0, 500, 100))
if portal_actor:
    portal_actor.set_actor_label("Portal_To_Sphere")
    mesh_comp = portal_actor.static_mesh_component
    mesh = unreal.EditorAssetLibrary.load_asset('/Engine/BasicShapes/Plane')
    mesh_comp.set_static_mesh(mesh)
    portal_actor.set_actor_scale3d(unreal.Vector(2, 2, 2))
    portal_actor.set_actor_rotation(unreal.Rotator(90,0,0), False)

# Save the level so it isn't lost again
level_subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
current_world = level_subsystem.get_current_level()
if current_world:
    unreal.EditorLoadingAndSavingUtils.save_dirty_packages(True, True)
    unreal.log("Vegas Sphere rebuilt and level saved!")
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
