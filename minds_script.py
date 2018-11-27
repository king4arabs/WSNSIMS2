from wsnsims.conductor.driver import run_minds, Parameters
from wsnsims.conductor import sim_inputs

parameters = [Parameters._make(p) for p in sim_inputs.conductor_params][0]

if __name__ == "__main__":
    results = run_minds(parameters)
    print(results)
