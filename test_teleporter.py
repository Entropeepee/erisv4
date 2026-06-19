import sys
import time
sys.path.append(r"C:\Program Files\Epic Games\UE_5.8\Engine\Plugins\Experimental\PythonScriptPlugin\Content\Python")
from remote_execution import RemoteExecution, MODE_EXEC_FILE

script = """
import unreal
actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)

# Spawn Teleporter at Fire
t1 = actor_subsystem.spawn_actor_from_class(unreal.Teleporter.static_class(), unreal.Vector(0,0,100))
if t1:
    t1.set_actor_label("Teleport_Fire")
    unreal.log("Spawned Teleporter 1")

# Spawn Teleporter at Desert
t2 = actor_subsystem.spawn_actor_from_class(unreal.Teleporter.static_class(), unreal.Vector(150000,0,100))
if t2:
    t2.set_actor_label("Teleport_Desert")
    unreal.log("Spawned Teleporter 2")

# Attempt to link them if there is a URL property
try:
    t1.set_editor_property('url', 'Teleport_Desert')
    t2.set_editor_property('url', 'Teleport_Fire')
except Exception as e:
    unreal.log_error(str(e))
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
