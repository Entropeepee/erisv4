import sys
import time
sys.path.append(r"C:\Program Files\Epic Games\UE_5.8\Engine\Plugins\Experimental\PythonScriptPlugin\Content\Python")
from remote_execution import RemoteExecution, MODE_EXEC_FILE

remote_exec = RemoteExecution()
remote_exec.start()
time.sleep(1.0)
nodes = remote_exec.remote_nodes
if nodes:
    node_id = nodes[0]['node_id']
    remote_exec.open_command_connection(node_id)
    unreal_script = """
import unreal
all_mats = unreal.EditorAssetLibrary.list_assets('/Game', recursive=True, include_folder=False)
mats = [m for m in all_mats if 'material' in m.lower() or 'mi_' in m.lower() or 'mat_' in m.lower() or 'm_' in m.lower()]
grass_mats = [m for m in mats if 'grass' in m.lower() or 'dirt' in m.lower() or 'ground' in m.lower() or 'landscape' in m.lower()]
print("GRASS MATS: ", grass_mats)
"""
    res = remote_exec.run_command(unreal_script, exec_mode=MODE_EXEC_FILE)
    print(res)
    remote_exec.stop()
