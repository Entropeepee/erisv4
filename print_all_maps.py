import sys
import time
sys.path.append(r"C:\Program Files\Epic Games\UE_5.8\Engine\Plugins\Experimental\PythonScriptPlugin\Content\Python")
from remote_execution import RemoteExecution, MODE_EXEC_FILE

script = """
import unreal

asset_reg = unreal.AssetRegistryHelpers.get_asset_registry()
all_maps = asset_reg.get_assets_by_path('/Game', recursive=True)
for m in all_maps:
    if m.asset_class_path.package_name == '/Script/Engine' and m.asset_class_path.asset_name == 'World':
        unreal.log(f"Map: {m.get_full_name()}")
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
