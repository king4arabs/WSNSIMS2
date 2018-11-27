from wsnsims.conductor.driver import run_loaf, Parameters
from wsnsims.conductor import sim_inputs

parameters = [Parameters._make(p) for p in sim_inputs.conductor_params][0]

if __name__ == "__main__":
    results = run_loaf(parameters)
    print(results)
