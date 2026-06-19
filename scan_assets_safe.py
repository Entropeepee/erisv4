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
all_assets = unreal.EditorAssetLibrary.list_assets('/Game', recursive=True, include_folder=False)
elderboom_assets = [a for a in all_assets if 'elderboom' in a.lower()]
eris_assets = [a for a in all_assets if 'natalia' in a.lower() or 'eris' in a.lower()]
quixel_assets = [a for a in all_assets if 'megascans' in a.lower()]
print(f"Elderboom: {len(elderboom_assets)}")
print(f"Sample Elderboom: {elderboom_assets[:10]}")
print(f"Eris: {len(eris_assets)}")
print(f"Sample Eris: {eris_assets[:10]}")
print(f"Quixel: {len(quixel_assets)}")
print(f"Sample Quixel: {quixel_assets[:10]}")
"""
    res = remote_exec.run_command(unreal_script, exec_mode=MODE_EXEC_FILE)
    print(res)
    remote_exec.stop()
