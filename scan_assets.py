import sys
import time
sys.path.append(r"C:\Program Files\Epic Games\UE_5.8\Engine\Plugins\Experimental\PythonScriptPlugin\Content\Python")
from remote_execution import RemoteExecution, MODE_EXEC_FILE
import json

remote_exec = RemoteExecution()
remote_exec.start()
time.sleep(1.0)

nodes = remote_exec.remote_nodes
if not nodes:
    print("No nodes found")
    sys.exit(1)

node_id = nodes[0]['node_id']
remote_exec.open_command_connection(node_id)

unreal_script = """
import unreal
import json

asset_reg = unreal.AssetRegistryHelpers.get_asset_registry()

def get_assets_by_class(class_name):
    assets = asset_reg.get_assets_by_class(class_name, True)
    return [str(a.package_name) for a in assets]

# Look for blueprints (to find Natalia/Eris)
bp_assets = get_assets_by_class('Blueprint')
eris_bps = [bp for bp in bp_assets if 'natalia' in bp.lower() or 'eris' in bp.lower()]

# Look for maps (to stitch scenes)
map_assets = get_assets_by_class('World')

# Look for Megascans or specific static meshes
sm_assets = get_assets_by_class('StaticMesh')
quixel_assets = [sm for sm in sm_assets if 'megascans' in sm.lower() or 'ms_' in sm.lower()]

res = {
    'eris_blueprints': eris_bps,
    'maps': [m for m in map_assets if 'thirdperson' not in m.lower() and 'firstperson' not in m.lower()][:20], # limit to 20
    'quixel_count': len(quixel_assets),
    'quixel_samples': quixel_assets[:10]
}

print("JSON_START" + json.dumps(res) + "JSON_END")
"""

res = remote_exec.run_command(unreal_script, exec_mode=MODE_EXEC_FILE)
print("Output:", res)
remote_exec.stop()
