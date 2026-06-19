import sys
import time
sys.path.append(r"C:\Program Files\Epic Games\UE_5.8\Engine\Plugins\Experimental\PythonScriptPlugin\Content\Python")
from remote_execution import RemoteExecution, MODE_EXEC_FILE

script = """
import unreal

actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)

# 1. The Vegas Sphere
# Position it high up so it's a massive floating orb
sphere_loc = unreal.Vector(-10000, 10000, 5000)
sphere_actor = actor_subsystem.spawn_actor_from_class(unreal.StaticMeshActor.static_class(), sphere_loc)
sphere_actor.set_actor_label("VegasSphere_Outer")
sphere_actor.set_actor_scale3d(unreal.Vector(100, 100, 100))
sphere_mesh = unreal.EditorAssetLibrary.load_asset('/Engine/BasicShapes/Sphere')
sphere_actor.static_mesh_component.set_static_mesh(sphere_mesh)
sphere_actor.static_mesh_component.set_collision_profile_name("NoCollision")

# We need a Two-Sided or inverted material. For now, we will leave the default material 
# and wait for the starry night toggle script to map the UI.

# 2. The Internal Platform
platform_loc = unreal.Vector(-10000, 10000, 1000)
platform_actor = actor_subsystem.spawn_actor_from_class(unreal.StaticMeshActor.static_class(), platform_loc)
platform_actor.set_actor_label("VegasSphere_Platform")
platform_actor.set_actor_scale3d(unreal.Vector(30, 30, 0.5))
cyl_mesh = unreal.EditorAssetLibrary.load_asset('/Engine/BasicShapes/Cylinder')
platform_actor.static_mesh_component.set_static_mesh(cyl_mesh)

# 3. The Portal (in the Garden)
portal_loc = unreal.Vector(0, 0, 50)
portal_actor = actor_subsystem.spawn_actor_from_class(unreal.StaticMeshActor.static_class(), portal_loc)
portal_actor.set_actor_label("Portal_To_Sphere")
portal_actor.set_actor_scale3d(unreal.Vector(1, 4, 6))
box_mesh = unreal.EditorAssetLibrary.load_asset('/Engine/BasicShapes/Cube')
portal_actor.static_mesh_component.set_static_mesh(box_mesh)

unreal.log("Vegas Sphere and Portal constructed.")
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
    print("Could not connect to Unreal Engine. It may still be frozen.")
