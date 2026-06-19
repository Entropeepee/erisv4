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

# 1. Stitch ONLY Dream World Garden (Lightweight!)
garden_asset = unreal.EditorAssetLibrary.load_asset('/Game/DreamWorld_Garden')
if garden_asset:
    garden_inst = spawn_if_not_exists(unreal.LevelInstance.static_class(), unreal.Vector(0, 150000, 0), "Biome_DreamWorldGarden")
    if garden_inst:
        garden_inst.set_editor_property('world_asset', garden_asset)
        unreal.log("Stitched Dream World Garden at Y=150000")

# 2. Spawn Teleport Trigger & Target
garden_target = spawn_if_not_exists(unreal.TargetPoint.static_class(), unreal.Vector(0, 150000, 500), "TeleportTarget_Garden")
fire_trigger = spawn_if_not_exists(unreal.TriggerBox.static_class(), unreal.Vector(0, 0, 100), "Fire_Teleporter_Trigger")
if fire_trigger:
    fire_trigger.set_actor_scale3d(unreal.Vector(2, 2, 2))

# 3. Spawn Dream World Environment
sky_atm = spawn_if_not_exists(unreal.SkyAtmosphere.static_class(), unreal.Vector(0, 150000, 0), "DreamWorld_SkyAtmosphere")
clouds = spawn_if_not_exists(unreal.VolumetricCloud.static_class(), unreal.Vector(0, 150000, 0), "DreamWorld_VolumetricClouds")
sun = spawn_if_not_exists(unreal.DirectionalLight.static_class(), unreal.Vector(0, 150000, 5000), "DreamWorld_Sun")
if sun:
    sun.set_actor_rotation(unreal.Rotator(0, 45, -45), False)
    sun_comp = sun.get_component_by_class(unreal.DirectionalLightComponent.static_class())
    if sun_comp:
        try: sun_comp.set_editor_property("atmosphere_sun_light", True)
        except: pass

fog = spawn_if_not_exists(unreal.ExponentialHeightFog.static_class(), unreal.Vector(0, 150000, 0), "DreamWorld_Fog")
if fog:
    fog_comp = fog.get_component_by_class(unreal.ExponentialHeightFogComponent.static_class())
    if fog_comp:
        try:
            fog_comp.set_editor_property("volumetric_fog", True)
            fog_comp.set_editor_property("fog_density", 0.05)
        except: pass

lake = spawn_if_not_exists(unreal.StaticMeshActor.static_class(), unreal.Vector(2000, 152000, 50), "DreamWorld_LakePlaceholder")
if lake:
    lake_comp = lake.get_component_by_class(unreal.StaticMeshComponent.static_class())
    plane_mesh = unreal.EditorAssetLibrary.load_asset("/Engine/BasicShapes/Plane")
    if plane_mesh and lake_comp:
        lake_comp.set_static_mesh(plane_mesh)
        lake.set_actor_scale3d(unreal.Vector(100, 100, 1))
        glass_mat = unreal.EditorAssetLibrary.load_asset("/Engine/MapTemplates/Materials/BasicAsset01")
        if glass_mat:
            lake_comp.set_material(0, glass_mat)

ambient_wind = spawn_if_not_exists(unreal.AmbientSound.static_class(), unreal.Vector(0, 150000, 500), "DreamWorld_Sound_WindRain")

unreal.EditorLoadingAndSavingUtils.save_dirty_packages(True, True)
unreal.log("Garden environment spawned securely and saved!")
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
