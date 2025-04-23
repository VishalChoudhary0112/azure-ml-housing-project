# run_pipeline.py  – train + deploy + delete (credit‑safe)
from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.model import Model, InferenceConfig
from azureml.core.webservice import AciWebservice

# 0. Connect to current workspace (assumes config.json auto‑generated)
ws = Workspace.from_config()

# 1. Ensure cluster with autoscale‑to‑zero
cluster_name = "cpu-cluster"
if cluster_name in ws.compute_targets:
    cluster = ws.compute_targets[cluster_name]
else:
    cluster = ComputeTarget.create(
        ws, cluster_name,
        AmlCompute.provisioning_configuration(
            vm_size="STANDARD_DS11_V2",
            min_nodes=0, max_nodes=2))
    cluster.wait_for_completion(show_output=True)

# 2. Submit training job
env = Environment.from_conda_specification(
        name="housing-env", file_path="environment.yml")

src = ScriptRunConfig(source_directory="scripts",
                      script="train.py",
                      compute_target=cluster,
                      environment=env)

run = Experiment(ws, "housing-free").submit(src)
run.wait_for_completion(show_output=True)

# 3. Register model
model = run.register_model("housing_model", "outputs/model.pkl")

# 4. Deploy to ACI
inf_cfg = InferenceConfig(entry_script="score.py", environment=env)
aci_cfg = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

service = Model.deploy(ws, "housing-free-aci", [model],
                       inf_cfg, aci_cfg, overwrite=True)
service.wait_for_deployment(show_output=True)
print("\n🎯 Endpoint:", service.scoring_uri)

input("\nPress Enter after testing to DELETE the endpoint and save cost → ")
service.delete()
print("ACI deleted ✅")
