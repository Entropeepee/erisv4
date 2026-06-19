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

try:
    # Attempt to create a new level
    level_path = '/Game/DreamWorld_Garden'
    
    if unreal.EditorAssetLibrary.does_asset_exist(level_path):
        unreal.EditorLevelLibrary.load_level(level_path)
    else:
        unreal.EditorLevelLibrary.new_level(level_path)
    
    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    
    # Floor
    cube_mesh = unreal.EditorAssetLibrary.load_asset('/Engine/BasicShapes/Cube')
    floor = actor_subsystem.spawn_actor_from_class(unreal.StaticMeshActor.static_class(), unreal.Vector(0,0,-50))
    if floor:
        floor.static_mesh_component.set_static_mesh(cube_mesh)
        floor.set_actor_scale3d(unreal.Vector(200, 200, 1))
        floor.set_actor_label("Garden_Floor")
    
    # Lighting
    dir_light = actor_subsystem.spawn_actor_from_class(unreal.DirectionalLight.static_class(), unreal.Vector(0,0,1000))
    if dir_light:
        dir_light.set_actor_label("SunLight")
    
    sky_atm = actor_subsystem.spawn_actor_from_class(unreal.SkyAtmosphere.static_class(), unreal.Vector(0,0,0))
    sky_light = actor_subsystem.spawn_actor_from_class(unreal.SkyLight.static_class(), unreal.Vector(0,0,0))
    
    # Player Start
    player_start = actor_subsystem.spawn_actor_from_class(unreal.PlayerStart.static_class(), unreal.Vector(0,0,100))
    if player_start:
        player_start.set_actor_label("Main_PlayerStart")
    
    # Visualizer Room (open top box)
    wall_scale = unreal.Vector(20, 1, 10)
    room_center = unreal.Vector(3000, 0, 450)
    
    w1 = actor_subsystem.spawn_actor_from_class(unreal.StaticMeshActor.static_class(), room_center + unreal.Vector(0, 1000, 0))
    w1.static_mesh_component.set_static_mesh(cube_mesh)
    w1.set_actor_scale3d(wall_scale)
    w1.set_actor_label("Visualizer_Wall_North")
    
    w2 = actor_subsystem.spawn_actor_from_class(unreal.StaticMeshActor.static_class(), room_center + unreal.Vector(0, -1000, 0))
    w2.static_mesh_component.set_static_mesh(cube_mesh)
    w2.set_actor_scale3d(wall_scale)
    w2.set_actor_label("Visualizer_Wall_South")
    
    w3 = actor_subsystem.spawn_actor_from_class(unreal.StaticMeshActor.static_class(), room_center + unreal.Vector(1000, 0, 0))
    w3.static_mesh_component.set_static_mesh(cube_mesh)
    w3.set_actor_scale3d(unreal.Vector(1, 20, 10))
    w3.set_actor_label("Visualizer_Wall_East")
    
    w4 = actor_subsystem.spawn_actor_from_class(unreal.StaticMeshActor.static_class(), room_center + unreal.Vector(-1000, 0, 0))
    w4.static_mesh_component.set_static_mesh(cube_mesh)
    w4.set_actor_scale3d(unreal.Vector(1, 20, 10))
    w4.set_actor_label("Visualizer_Wall_West")
    
    # Sphere Room
    sphere_mesh = unreal.EditorAssetLibrary.load_asset('/Engine/BasicShapes/Sphere')
    sphere_room = actor_subsystem.spawn_actor_from_class(unreal.StaticMeshActor.static_class(), unreal.Vector(-3000, 0, 2000))
    if sphere_room:
        sphere_room.static_mesh_component.set_static_mesh(sphere_mesh)
        sphere_room.set_actor_scale3d(unreal.Vector(40, 40, 40))
        sphere_room.set_actor_label("Vegas_Sphere_Room")
    
    unreal.EditorLevelLibrary.save_current_level()
    print("SUCCESS: Dream World created.")
except Exception as e:
    print(f"ERROR: {e}")
"""

res = remote_exec.run_command(unreal_script, exec_mode=MODE_EXEC_FILE)
print("Execution result:", res)
remote_exec.stop()
