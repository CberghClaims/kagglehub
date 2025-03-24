# Import necessary libraries
import pandas as pd
from datetime import timedelta
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.distance import geodesic
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# -----------------------------
# STEP 1: Load your data
# -----------------------------
# Assuming your Excel file has a column named 'Address'
data_file = 'Claims.xlsx'
df = pd.read_excel(data_file)

# Convert addresses to a list
inspection_addresses = df['Address'].tolist()

# -----------------------------
# STEP 2: Add the depot address
# -----------------------------
depot_address = "251 Buffalo Lane, Tazewell, TN 37879"
# Place depot at the start of the list
addresses = [depot_address] + inspection_addresses
# Also, ensure the depot is the end point (for round trip, OR-Tools expects same start and end)

# -----------------------------
# STEP 3: Geocode addresses to obtain coordinates
# -----------------------------
geolocator = Nominatim(user_agent="statefarm_route_optimizer")
# To avoid exceeding rate limits
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

# Dictionary to hold address: (lat, lon)
address_coords = {}
for addr in addresses:
    location = geocode(addr)
    if location is None:
        raise ValueError(f"Geocoding failed for address: {addr}")
    address_coords[addr] = (location.latitude, location.longitude)

# Create a list of coordinates in the same order as addresses
coords = [address_coords[addr] for addr in addresses]

# -----------------------------
# STEP 4: Build the travel time matrix
# -----------------------------
# We calculate travel time based on geodesic distance.
# Assume an average driving speed (e.g., 40 mph)
avg_speed_mph = 40
# Conversion: miles / speed * 60 = travel time in minutes

def compute_travel_time(coord1, coord2):
    # Get distance in miles
    distance = geodesic(coord1, coord2).miles
    travel_time = (distance / avg_speed_mph) * 60
    return int(travel_time)

n = len(coords)
time_matrix = [[0] * n for _ in range(n)]
for i in range(n):
    for j in range(n):
        if i != j:
            time_matrix[i][j] = compute_travel_time(coords[i], coords[j])
        else:
            time_matrix[i][j] = 0

# -----------------------------
# STEP 5: Set service time and time windows
# -----------------------------
service_time = 30  # minutes per inspection

# Time windows: For the depot, we start at 9:00 (540) and must return by 16:00 (960)
# For inspections, arrival time must be such that service can be completed by 960.
time_windows = []
# Depot time window
time_windows.append((540, 960))
# For each inspection (assumed same window for simplicity)
for _ in inspection_addresses:
    # For an inspection, the window is also [540, 960-service_time] so that the service finishes by 960.
    time_windows.append((540, 960 - service_time))

# -----------------------------
# STEP 6: Create the OR-Tools Routing Model
# -----------------------------
def create_data_model():
    data = {}
    data['time_matrix'] = time_matrix
    data['time_windows'] = time_windows
    data['num_vehicles'] = 1  # one vehicle (your day)
    data['depot'] = 0
    data['service_time'] = service_time
    return data

data = create_data_model()

# Create the routing index manager.
manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']),
                                       data['num_vehicles'], data['depot'])
# Create Routing Model.
routing = pywrapcp.RoutingModel(manager)

# Define cost of each arc.
def time_callback(from_index, to_index):
    # Convert routing variable indices to time matrix node indices.
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return data['time_matrix'][from_node][to_node] + data['service_time']

transit_callback_index = routing.RegisterTransitCallback(time_callback)
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

# Add Time Window constraints.
routing.AddDimension(
    transit_callback_index,
    30,  # allow waiting time
    960,  # maximum time per vehicle
    False,  # Don't force start cumul to zero. We have a defined start time.
    'Time')
time_dimension = routing.GetDimensionOrDie('Time')

# Apply time window constraints for each location.
for location_idx, time_window in enumerate(data['time_windows']):
    index = manager.NodeToIndex(location_idx)
    time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])

# Ensure that the route starts at 9:00.
start_index = routing.Start(0)
time_dimension.CumulVar(start_index).SetValue(540)

# -----------------------------
# STEP 7: Solve the problem
# -----------------------------
search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PathCheapestArc)
search_parameters.time_limit.seconds = 30

solution = routing.SolveWithParameters(search_parameters)

if solution:
    # Retrieve the route and schedule.
    index = routing.Start(0)
    plan_output = []
    total_time = 0
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        time_var = time_dimension.CumulVar(index)
        plan_output.append({
            'Stop': addresses[node],
            'Arrival': solution.Min(time_var)
        })
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        total_time += routing.GetArcCostForVehicle(previous_index, index, 0)
    # Add the depot as the final stop
    node = manager.IndexToNode(index)
    time_var = time_dimension.CumulVar(index)
    plan_output.append({
        'Stop': addresses[node],
        'Arrival': solution.Min(time_var)
    })

    # Convert minutes to time strings (assuming day starts at midnight)
    def minutes_to_time_str(minutes):
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours:02d}:{mins:02d}"

    print("Optimized Route and Schedule:")
    for stop in plan_output:
        print(f"Stop: {stop['Stop']} - Arrival Time: {minutes_to_time_str(stop['Arrival'])}")
    print(f"Total route time (including service and travel): {total_time} minutes")
else:
    print("No solution found within the given time window constraints.")
