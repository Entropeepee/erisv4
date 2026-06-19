import sys
import time
sys.path.append(r"C:\Program Files\Epic Games\UE_5.8\Engine\Plugins\Experimental\PythonScriptPlugin\Content\Python")
from remote_execution import RemoteExecution, MODE_EXEC_FILE

script = """
import unreal
unreal.EditorAssetLibrary.delete_asset('/Game/Core/Blueprints/UI/WB_Settings')
unreal.EditorAssetLibrary.delete_asset('/Game/Core/Blueprints/UI/WB_Components/WB_A2F_ParamOverrides')
unreal.log("Deleted broken Audio2Face assets.")
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
