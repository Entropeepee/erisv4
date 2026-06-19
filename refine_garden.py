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
    
    all_actors = actor_subsystem.get_all_level_actors()
    elderboom_inst = None
    floor = None
    
    for a in all_actors:
        label = a.get_actor_label()
        if "Elderboom" in label and a.get_class() == unreal.LevelInstance.static_class():
            elderboom_inst = a
        elif "Garden_Floor" in label:
            floor = a

    if elderboom_inst:
        # Move it far away
        new_loc = unreal.Vector(15000, 15000, -50)
        elderboom_inst.set_actor_location(new_loc, False, False)
        unreal.log("Moved Elderboom to a clear area.")
        
        # Add Post Process Volume around Elderboom for moodier lighting
        pp_vol = actor_subsystem.spawn_actor_from_class(unreal.PostProcessVolume.static_class(), new_loc)
        if pp_vol:
            pp_vol.set_actor_label("Elderboom_Mood_PPV")
            # Scale bounds to cover the village (20000x20000x10000 roughly)
            pp_vol.set_actor_scale3d(unreal.Vector(200, 200, 100))
            
            settings = pp_vol.settings
            settings.bOverride_AutoExposureMinBrightness = True
            settings.auto_exposure_min_brightness = 0.5
            settings.bOverride_AutoExposureMaxBrightness = True
            settings.auto_exposure_max_brightness = 1.0
            
            settings.bOverride_ColorSaturation = True
            settings.color_saturation = unreal.Vector4(0.8, 0.8, 0.9, 1.0)
            
            pp_vol.settings = settings
            unreal.log("Added Post Process Volume for Elderboom.")

    if floor:
        mat_asset = unreal.EditorAssetLibrary.load_asset('/Game/ElderboomVillage/Materials/MI_Grass_Clumps_rbojr_2K')
        if mat_asset:
            floor.static_mesh_component.set_material(0, mat_asset)
            unreal.log("Applied grass material to the garden floor.")
            
    unreal.EditorLevelLibrary.save_current_level()
    print("SUCCESS: Garden refined.")
except Exception as e:
    print(f"ERROR: {e}")
"""

res = remote_exec.run_command(unreal_script, exec_mode=MODE_EXEC_FILE)
print("Execution result:", res)
remote_exec.stop()
