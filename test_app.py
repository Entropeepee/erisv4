import sys
import trace
print('Starting import')
tracer = trace.Trace(count=False, trace=True, ignoredirs=[sys.prefix, sys.exec_prefix])

def run():
    from eris.server.app import create_app
    app = create_app()
    print('App created')

tracer.run('run()')
print('Done')
