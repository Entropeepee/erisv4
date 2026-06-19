import sys
import time
sys.path.append(r"C:\Program Files\Epic Games\UE_5.8\Engine\Plugins\Experimental\PythonScriptPlugin\Content\Python")
from remote_execution import RemoteExecution, MODE_EXEC_FILE

script = """
import unreal

# Duplicate the asset
asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
if unreal.EditorAssetLibrary.does_asset_exist('/Game/WBP_Pedestal'):
    if not unreal.EditorAssetLibrary.does_asset_exist('/Game/WBP_WallScreen'):
        unreal.EditorAssetLibrary.duplicate_asset('/Game/WBP_Pedestal', '/Game/WBP_WallScreen')
        unreal.log("Duplicated WBP_Pedestal into WBP_WallScreen.")
    
    # Spawn Pedestal into level
    pedestal_class = unreal.EditorAssetLibrary.load_blueprint_class('/Game/WBP_Pedestal')
    if pedestal_class:
        # Widget placement
        loc = unreal.Vector(-10000, 10000, 1050)
        # Wait, you cannot spawn_actor_from_class on a UserWidget class, it must be a WidgetActor or we spawn a basic Actor and add component.
        # Let's see if spawn_actor_from_object works for UserWidget blueprint
        bp = unreal.EditorAssetLibrary.load_asset('/Game/WBP_Pedestal')
        actor = unreal.EditorLevelLibrary.spawn_actor_from_object(bp, loc)
        if actor:
            actor.set_actor_label("Pedestal_Console")
            unreal.log("Spawned Pedestal Actor.")
        else:
            unreal.log_warning("Could not spawn Widget Actor automatically.")
else:
    unreal.log_error("Could not find /Game/WBP_Pedestal. Make sure it is saved in the root Content folder.")
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
    print("Could not connect to Unreal Engine.")
