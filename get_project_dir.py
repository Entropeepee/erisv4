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
print(unreal.Paths.project_dir())
"""

res = remote_exec.run_command(unreal_script, exec_mode=MODE_EXEC_FILE)
print("Execution result:", res)
remote_exec.stop()
