import os
import traci
import time 

def run():
    if "SUMO_HOME" not in os.environ:
        raise RuntimeError("SUMO_HOME not set you fool")

    sumo_binary = os.path.join(os.environ["SUMO_HOME"], "bin", "sumo-gui")
    sumo_cfg = "simple.sumocfg"

    # Start SUMO under TraCI control
    traci.start([sumo_binary, "-c", sumo_cfg, "--start", "--no-step-log", "true"])

    tls_ids = traci.trafficlight.getIDList()
    print("Traffic lights:", tls_ids)
    tls_id = tls_ids[0]

    # Try to inspect the traffic light program and phases
    n_phases = None
    try:
        logic_list = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)
        if logic_list:
            logic = logic_list[0]
            n_phases = len(logic.phases)
            print(f"Program ID: {logic.programID}, number of phases: {n_phases}")
            for i, p in enumerate(logic.phases):
                print(f"  Phase {i}: duration={p.duration}, state={p.state}")
    except Exception as e:
        print("Could not inspect phases:", e)

    step = 0
    max_steps = 200

    incoming_edges = ["north_in", "south_in", "east_in", "west_in"]

    while step < max_steps:
        traci.simulationStep()
        time.sleep(0.1)

        # Observation: queue length on each incoming edge
        queues = [
            traci.edge.getLastStepHaltingNumber(e)
            for e in incoming_edges
        ]
        print(f"Step {step:3d} queues: {queues}")

        if step % 60 < 30:
            traci.trafficlight.setPhase(tls_id, 0)
        else:
            if n_phases is not None:
                if n_phases > 2:
                    traci.trafficlight.setPhase(tls_id, 2)
                elif n_phases > 1:
                    traci.trafficlight.setPhase(tls_id, 1)
                else:
                    traci.trafficlight.setPhase(tls_id, 0)
            else:
                traci.trafficlight.setPhase(tls_id, 0)

        step += 1

    traci.close()
    print("Simulation finished.")

if __name__ == "__main__":
    run()
