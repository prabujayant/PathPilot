import json
import time
import random
import os

STATS_FILE = "pox_stats.json"
ACTIONS_FILE = "qos_actions.json"

# Switches and ports logic matches visualizer.html
# s1(dpid 1): p1(h1), p2(s4) (and maybe p3 connected to something else? Visualizer only shows these)
# s2(dpid 2): p1(h1), p2(s4)
# s3(dpid 3): p1(h1), p2(s4)
# s4(dpid 4): p1(s1?), p2(s2?), p3(s3?), p4(h2) 
# Note: Visualizer reads stats[dpid][port].

def generate_base_stats():
    return {
        "tx_bps": random.uniform(0, 1000),
        "rx_bps": random.uniform(0, 1000),
        "tx_packets": 0,
        "rx_packets": 0,
        "tx_dropped": 0
    }

def update_traffic(current_traffic):
    # Fluoresce traffic
    change = random.uniform(-2e6, 2e6) # +/- 2 Mbps
    new_val = current_traffic + change
    return max(0, min(new_val, 150e6)) # Cap at 150 Mbps

traffic_state = {
    "1": 5e6, # s1 path
    "2": 5e6, # s2 path
    "3": 5e6  # s3 path
}

def main():
    print("Starting Mock Traffic Simulation...")
    
    # Init empty actions if not exists
    if not os.path.exists(ACTIONS_FILE):
        with open(ACTIONS_FILE, "w") as f:
            json.dump([], f)

    while True:
        # Check control
        try:
            with open("traffic_control.json", "r") as f:
                control = json.load(f)
                active = control.get("active", True)
        except:
            active = True # default on

        if not active:
            # Silence traffic
            for k in traffic_state:
                traffic_state[k] = 0
            
            # Still update stats file
            stats = {
                "1": {"1": {"tx_bps": 0, "rx_bps": 0}, "2": {"tx_bps": 0, "rx_bps": 0}},
                "2": {"1": {"tx_bps": 0, "rx_bps": 0}, "2": {"tx_bps": 0, "rx_bps": 0}},
                "3": {"1": {"tx_bps": 0, "rx_bps": 0}, "2": {"tx_bps": 0, "rx_bps": 0}},
                "4": {"4": {"tx_bps": 0, "rx_bps": 0}}
            }
             # Write stats
            with open(STATS_FILE, "w") as f:
                json.dump(stats, f)
            time.sleep(1)
            continue


        # Update flows
        # Simulate load shifting
        if random.random() < 0.4:
            # Randomly spike one path
            target = random.choice(["1", "2", "3"])
            traffic_state[target] += 100e6
            print(f"Computed Spike on Path {target}")

        # Decay high traffic
        for k in traffic_state:
            traffic_state[k] *= 0.95
            if traffic_state[k] < 1e5: traffic_state[k] = random.uniform(0, 1e5)

        stats = {
            "1": {
                "1": {"tx_bps": traffic_state["1"], "rx_bps": traffic_state["1"]}, 
                "2": {"tx_bps": traffic_state["1"], "rx_bps": traffic_state["1"]}
            },
            "2": {
                "1": {"tx_bps": traffic_state["2"], "rx_bps": traffic_state["2"]},
                "2": {"tx_bps": traffic_state["2"], "rx_bps": traffic_state["2"]}
            },
            "3": {
                "1": {"tx_bps": traffic_state["3"], "rx_bps": traffic_state["3"]},
                "2": {"tx_bps": traffic_state["3"], "rx_bps": traffic_state["3"]}
            },
            "4": {
                "4": {"tx_bps": sum(traffic_state.values()), "rx_bps": sum(traffic_state.values())} # Aggregate at h2
            }
        }

        # Simulate drops if congestion > 80Mbps
        for sw in ["1", "2", "3"]:
            if traffic_state[sw] > 80e6:
                stats[sw]["2"]["tx_dropped"] = int((traffic_state[sw] - 80e6) / 10000)
                
                # Maybe trigger an action
                if random.random() < 0.8:
                    action = {
                        "type": "REROUTE",
                        "message": f"Congestion on Switch {sw}. Rerouting flow.",
                        "timestamp": time.time()
                    }
                    try:
                        with open(ACTIONS_FILE, "r") as f:
                            actions = json.load(f)
                        actions.append(action)
                        if len(actions) > 20: actions = actions[-20:]
                        with open(ACTIONS_FILE, "w") as f:
                            json.dump(actions, f)
                    except Exception as e:
                        print("Error writing actions:", e)

        # Write stats
        with open(STATS_FILE, "w") as f:
            json.dump(stats, f)
        
        time.sleep(1)

if __name__ == "__main__":
    main()
