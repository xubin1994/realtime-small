import rrun

client = rrun.Client()

spec = rrun.RunnerSpec()
spec.name = 'my-runner'
spec.commands[:] = ['bash', '-c', 'ulimit -n 65535; python Train_dispnet.py']
spec.log_dir = '/home/jiangying/logs'
spec.scheduling_hint.group = '' # Inherit from master
spec.resources.cpu = 2
spec.resources.gpu = 1
spec.enable_ssh = True
spec.resources.memory_in_mb = 20480
spec.preemptible = False
spec.priority = 'Medium' # Default: 'Medium'. Change this in need
spec.max_wait_time = 3600 * int(1e9)
spec.minimum_lifetime = 24 * 3600 * int(1e9)
spec.capabilities[:] = [rrun.RunnerSpec.SYS_PTRACE] # Default: empty

# Fill in several fields in runner spec from current environment.
rrun.fill_runner_spec(
    spec,
    environments=True, # Propagate environment variables to runner
    uid_gid=True, # Propagate linux uid and gids to runner
    share_dirs=True, # Propagate writability of share directories to runner
    work_dir=True, # Propagate current working directory to runner
)

response = client.start_runner(rrun.StartRunnerRequest(spec=spec))
runner_id = response.id
print(runner_id)
