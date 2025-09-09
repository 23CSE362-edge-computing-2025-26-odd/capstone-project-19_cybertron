import json
from collections import defaultdict

class Tier3Cloud:
    def __init__(self):
        # (1) Data Aggregation
        self.data_lake = []

        # (4) Fleet Monitoring
        self.fleet_status = defaultdict(dict)

        # (5) Cross-Vehicle Collaboration
        self.hazard_events = []

    # --------------------------------------------------
    # (1) Data Aggregation
    # Called by Tier-2 when it forwards cleaned logs
    def collect_data(self, vehicle_id, log):
        self.data_lake.append({"vehicle_id": vehicle_id, "log": log})
        print(f"[Tier-3] Collected log from {vehicle_id}: {log}")

    # --------------------------------------------------
    # (4) Fleet Monitoring
    # Tier-2 forwards vehicle status here
    def update_fleet_status(self, vehicle_id, status):
        """
        status example:
        {
          "battery": %, "speed": km/h,
          "location": ("Zone-1", (lat,lon)), "event": str
        }
        """
        self.fleet_status[vehicle_id] = status
        print(f"[Tier-3] Updated fleet status for {vehicle_id}: {status}")

    def generate_fleet_report(self):
        """Summarize fleet-level statistics"""
        low_battery = [vid for vid, s in self.fleet_status.items()
                       if s.get("battery", 100) < 20]
        avg_speed = sum(s.get("speed", 0) for s in self.fleet_status.values()) / \
                    max(1, len(self.fleet_status))
        return {
            "total_vehicles": len(self.fleet_status),
            "low_battery_vehicles": low_battery,
            "average_speed": avg_speed
        }

    # --------------------------------------------------
    # (5) Cross-Vehicle Collaboration
    # Tier-2 forwards hazard info here
    def report_hazard(self, vehicle_id, hazard_type, location):
        event = {
            "vehicle_id": vehicle_id,
            "hazard": hazard_type,
            "location": location
        }
        self.hazard_events.append(event)
        print(f"[Tier-3] Hazard recorded: {hazard_type} in {location} by {vehicle_id}")

    def distribute_hazards(self, region=None):
        """Return hazard events to Tier-2 (to be sent to vehicles)"""
        if region:
            events = [e for e in self.hazard_events if e["location"] == region]
        else:
            events = self.hazard_events
        print(f"[Tier-3] Sharing {len(events)} hazards for region={region}")
        return events


# ======================================================
# === Demo (Simulating Tier-1 + Tier-2 calling Tier-3)
# ======================================================
if __name__ == "__main__":
    cloud = Tier3Cloud()

    # Simulated Tier-1 data forwarded by Tier-2
    cloud.collect_data("veh_A", {"event": "lane_change", "speed": 45})
    cloud.collect_data("veh_B", {"event": "obstacle_detected", "speed": 60})

    # Simulated fleet status updates from Tier-2
    cloud.update_fleet_status("veh_A", {"battery": 15, "speed": 45, "location": "Zone-1"})
    cloud.update_fleet_status("veh_B", {"battery": 80, "speed": 60, "location": "Zone-1"})

    # Generate fleet report
    print("\n=== Fleet Report ===")
    print(json.dumps(cloud.generate_fleet_report(), indent=2))

    # Hazards reported via Tier-2
    cloud.report_hazard("veh_A", "slippery_road", "Zone-1")
    cloud.report_hazard("veh_B", "accident", "Zone-1")

    # Distribution back to Tier-2 (so vehicles can be alerted)
    print("\n=== Hazard Distribution ===")
    print(json.dumps(cloud.distribute_hazards(region="Zone-1"), indent=2))
