import sys
import time
sys.path.append(r"C:\Program Files\Epic Games\UE_5.8\Engine\Plugins\Experimental\PythonScriptPlugin\Content\Python")
from remote_execution import RemoteExecution, MODE_EXEC_FILE
import json

remote_exec = RemoteExecution()
remote_exec.start()
time.sleep(1.0)
nodes = remote_exec.remote_nodes
if nodes:
    node_id = nodes[0]['node_id']
    remote_exec.open_command_connection(node_id)
    unreal_script = """
import unreal

# List all assets
all_assets = unreal.EditorAssetLibrary.list_assets('/Game', recursive=True, include_folder=False)

# Let's group them by base path to see what folders exist
folders = set()
for a in all_assets:
    parts = a.split('/')
    if len(parts) > 2:
        folders.add(parts[2])

print("TOP LEVEL FOLDERS IN /Game:", list(folders))

# Let's find static meshes that are NOT in Elderboom or MetaHumans
other_meshes = []
for a in all_assets:
    if 'ElderboomVillage' not in a and 'MetaHumans' not in a:
        if 'staticmesh' in a.lower() or 'sm_' in a.lower():
            other_meshes.append(a)

print(f"Found {len(other_meshes)} other meshes.")
print("SAMPLES:", other_meshes[:20])
"""
    res = remote_exec.run_command(unreal_script, exec_mode=MODE_EXEC_FILE)
    print(res)
    remote_exec.stop()
