"""
Script: Philly Vehicle Locator (PVL)
File Name: philly-vehicle-locator.py
Version: 1.0.2
Date Created: 10/24/2018
Author: Tim Haynes & Paul Sesink Clee
Last Update: 2/14/2019
Updater: Tim Haynes

Summary: Script for consuming NetworkFleet GPS points and adapting them for street segment outputs.

Credit: This script was adapted from 'mapmatching' simonscheider (https://github.com/simonscheider/mapmatching)
"""

# region Import libraries
import os
import sys
try:
    from ast import literal_eval
    from configparser import ConfigParser
    from datetime import datetime as dt, timedelta
    import arcpy
    import logging
    from math import exp, sqrt, radians, cos, sin, asin
    import networkx as nx
    import numpy
    import psycopg2
    import pytz
    import traceback
    import time
except ImportError:
    print('Import Error: Missing one or more libraries.')
    print(traceback.format_exc())
    sys.exit(1)
# endregion


def error_handler(log_string='Script failed.'):
    """Log errors as critical, including user input log string and traceback information. Print same information in
    console."""
    log.critical(log_string)
    log.critical(traceback.format_exc())
    print(log_string)
    print(traceback.format_exc())


def point_retriever(cursor, sql):
    """Retrieves points from hosted feature service containing gps points from GeoEvent Server & Network Fleet for
     previous config defined amount of time. Projects points into meter based projection in one of two ways depending on
     configuration. If default transformation method is acceptable, api_project should be set to True.  Otherwise, it
     must be set to False, with a transformation method provided."""
    cursor.execute(sql)
    current_vin = None
    incoming_points = {}
    point_list = cursor.fetchall()
    log.info('Number of points: {0}'.format(len(point_list)))
    if len(point_list) < 1:
        cursor.execute("SELECT MAX(fixtimeutf) FROM networkfleet_gps")
        last_point_time = production_database_cursor.fetchone()
        time_since_last_point = (scriptStart - int(last_point_time[0])) / 60.0
        error_handler('No points returned by query. Last point was {0} minutes ago.'.format(time_since_last_point))
        sys.exit(1)
    for gps_point in point_list:
        if not current_vin or current_vin != gps_point[0]:
            incoming_points[gps_point[0]] = [(gps_point[1], gps_point[2], gps_point[3])]
        else:
            if (gps_point[1], gps_point[2], gps_point[3]) not in incoming_points[gps_point[0]]:
                incoming_points[gps_point[0]].append((gps_point[1], gps_point[2], gps_point[3]))
            continue
        current_vin = gps_point[0]

    return incoming_points


def index_network_segment_info(input_network):
    """
    Builds a dictionary to be used as an index for looking up endpoints and lengths of network segments. Lengths are
    used by networkX to build the network graph. Endpoints are used later to test the connectivity of likely path
    segments.

    Parameters:
        input_network (shapefile projected in meters): Street network gps points should be matched to.

    Output: Returns two dictionaries. segment_end_points lists the geometry for the two endpoints of each street segment
     in input_network by seg_ID.  segment_lengths lists the length of each segment in input_network by seg_ID.
    """
    try:
        if os.path.isfile(str(os.path.join(arcpy.env.workspace, input_network))):
            segment_cursor = arcpy.da.SearchCursor(in_table=input_network, field_names=['SEG_ID', 'SHAPE@'])
            segment_end_points = {}
            segment_lengths = {}
            for segment in segment_cursor:
                segment_end_points[segment[0]] = ((segment[1].firstPoint.X, segment[1].firstPoint.Y),
                                                  (segment[1].lastPoint.X, segment[1].lastPoint.Y))
                segment_lengths[segment[0]] = segment[1].length
            del segment_cursor
            print('Number of segments in the input network: {0}'.format(len(segment_end_points)))
            return segment_end_points, segment_lengths
        else:
            raise FileNotFoundError('Street Network file not found. Make sure it exists and the config file is '
                                    'correct.')
    except FileNotFoundError as file_error:
        error_handler(str(file_error))


def create_network_graph(input_network, segment_lengths):
    """
    Uses the networkx library to build a network graph from the streets network segment input, including the length of
    the segment.

    Parameters:
        input_network (shapefile projected in meters): Street network gps points should be matched to.
        segment_lengths (dict): Output from index_network_segment_info function. Key=seg_id, Value=length of segment

    Output: subgraph: networkx graph of the street network segments
    """
    try:
        network_file_path = str(os.path.join(arcpy.env.workspace, input_network))
        if os.path.isfile(network_file_path):
            graph = nx.read_shp(network_file_path)
            # Select the largest connected component of the network graph.
            subgraph = list(nx.connected_component_subgraphs(graph.to_undirected()))[0]
            print('Graph size (excluding unconnected parts): {0}'.format(len(graph)))
            # Append the length of each road segment as an attribute to the edges in the network subgraph.
            for node_0, node_1 in subgraph.edges():
                segment_id = subgraph[node_0][node_1]['SEG_ID']
                subgraph[node_0][node_1]['length'] = segment_lengths[segment_id]
            return subgraph
        else:
            raise FileNotFoundError('Street Network file not found. Make sure it exists and the config file is '
                                    'correct.')
    except FileNotFoundError as file_error:
        error_handler(str(file_error))


def read_point_grid(grid):
    """
    Reads the point grid index into memory as a dictionary.

    Parameters:
        grid (feature class): Input grid index containing each potential gps point location to .0001 decimal degrees and
            each potential street segment candidate and probability pairing for each point

    Output: Returns point_grid_index as a dictionary of each potential gps point to .0001 decimal degrees, along with
        the candidate network street segments for each point. Key=gridkey, Value=candidate segments with probabilities
        (dictionary).  Using a dictionary to store the grid index has shown remarkably faster performance than using a
        search cursor on the original feature class.
    """
    try:
        if arcpy.Exists(dataset=grid):  # os.path.isfile(path) cannot read files in gdb, switch in future if possible
            point_grid_index = {}
            grid_cursor = arcpy.da.SearchCursor(grid, ['gridkey', 'candidate_segs'])
            for grid_point in grid_cursor:
                point_grid_index[grid_point[0]] = grid_point[1]
            del grid_cursor
            print('Point grid size: {0}'.format(len(point_grid_index)))
            return point_grid_index
        else:
            raise FileNotFoundError('Point grid file not found. Make sure it exists and the config file is correct.')
    except FileNotFoundError as file_error:
        error_handler(str(file_error))


def street_seg_identifier(gps_points, grid, net_decay_constant=30):
    # TODO carry config option for last point only through functions
    """
    This is the primary function in the script, inputting gps points and returning street segments on a solved path.

    Parameters:
        gps_points: GPS point track for a given vehicle (VIN)
        grid: Index grid of potential gps point locations across the city, including candidate segments and
            probabilities for each point
        net_decay_constant: Distance in meters after which the match probability falls under 0.34 (exponential decay),
            parameter depends on the intervals between successive points in the gps point track
    Output: Returns the street network segments and relevant attributes for the optimized solved path for the vehicle.
    """
    # Make sure constants are floats
    net_decay = float(net_decay_constant)

    # Initialize the path array to store
    path_array = [{}]

    # Read in points along with time and street network segment candidates for each
    gps_track = index_track_points(gps_points, grid)
    # TODO Write yards, etc directly into output table
    # If the GPS track has more than one point:
    if len(gps_track) > 1:
        # Initialize the first point
        segment_candidates = gps_track[0][1]
        for segment_candidate in segment_candidates:
            path_array[0][segment_candidate] = {'probability': gps_track[0][1][segment_candidate],
                                                'previous_segment': None, 'path': [], 'path_nodes': [],
                                                'time': gps_track[0][0]}
        # Run the Viterbi algorithm
        previous_segment_candidates = None
        for gps_point in range(1, len(gps_track)):
            gps_point_time = gps_track[gps_point][0]
            path_array.append({})
            # Store the segment candidates for the previous GPS point
            if not previous_segment_candidates:
                previous_segment_candidates = gps_track[0][1]
            else:
                previous_segment_candidates = gps_track[gps_point - 1][1]

            # Get the segment candidates for the current point along with their a-priori probabilities based on
            # Euclidean distance (these are calculated before time and stored in the point grid file for lookup
            for segment_candidate in gps_track[gps_point][1]:
                maximum_transition_probability = 0
                previous_street_segment = None
                path = []
                path_nodes = None
                # Determine the highest network transition probability from the previous point's candidates to the
                # current point's candidates and find the corresponding network path
                for previous_segment_candidate in previous_segment_candidates:
                    path_nodes = path_array[gps_point - 1][previous_segment_candidate]['path_nodes'][-10:]
                    network_transition_probability = calculate_network_transition_probability(
                        segment_1=previous_segment_candidate,
                        segment_2=segment_candidate, input_graph=network_graph, end_nodes=endpoints,
                        net_decay_constant=net_decay)
                    network_probability = network_transition_probability[0]
                    path_array_probability = path_array[gps_point - 1][previous_segment_candidate]['probability']
                    transition_probability = path_array_probability * network_probability

                    if transition_probability > maximum_transition_probability:
                        maximum_transition_probability = transition_probability
                        previous_street_segment = previous_segment_candidate
                        path = network_transition_probability[1]
                        if network_transition_probability[2] is not None:
                            path_nodes.append(network_transition_probability[2])

                maximum_probability = gps_track[gps_point][1][segment_candidate] * maximum_transition_probability
                path_array[gps_point][segment_candidate] = {'probability': maximum_probability,
                                                            'previous_segment': previous_street_segment,
                                                            'path': path, 'path_nodes': path_nodes,
                                                            'time': gps_point_time}
            maximum_value = max(value['probability'] for value in path_array[gps_point].values())
            maximum_value = (1 if maximum_value == 0 else maximum_value)
            for s in path_array[gps_point].keys():
                path_array[gps_point][s]['probability'] = path_array[gps_point][s]['probability'] / maximum_value
        solved_path = {}
        solved_path_index = 0
        maximum_end_track_probability = max(value['probability'] for value in path_array[-1].values())
        previous_state = None
        if maximum_end_track_probability == 0:
            raise ValueError('Point density error.')
        # Get the most probable ending state (street network segment) of the path and its previous state
        for state, data in path_array[-1].items():
            if data['probability'] == maximum_end_track_probability:
                state_point_time = data['time']
                solved_path[solved_path_index] = [state, state_point_time]
                solved_path_index += 1
                previous_state = state
                break
        # trigger = False
        # Follow the path until the first observed state to find the most probable states along the path and
        # corresponding sub-paths
        for path_index in range(len(path_array) - 2, -1, -1):
            try:
                probable_sub_path = path_array[path_index + 1][previous_state]['path']
                sub_path_time = path_array[path_index + 1][previous_state]['time']
                previous_state_segment = path_array[path_index + 1][previous_state]['previous_segment']
                if len(probable_sub_path) > 0:
                    for segment in probable_sub_path[::-1]:
                        sub_path_time -= 1  # TODO Interpolate instead of subtracting one second from next/previous
                        # point *TIME APPLIED TO SUBPATHS ONLY*
                        solved_path[solved_path_index] = [segment, sub_path_time]
                        solved_path_index += 1
                solved_path[solved_path_index] = [previous_state_segment,
                                                  path_array[path_index][previous_state_segment]['time']]
                solved_path_index += 1
                previous_state = previous_state_segment
            except KeyError:
                print('Key Error while finding path.')
                continue
        optimized_solved_path = optimize_solved_path(input_path=solved_path)
        return optimized_solved_path
    # Else if the GPS track has exactly one point, choose the segment candidate that has the highest probability and
    # write it directly to the output table
    elif len(gps_track) == 1:
        print('Found a gps_track with length == 1.')
        max_probable_segment = max(gps_track[0][1], key=lambda key: gps_track[0][1][key])
        print('Max probable segment is {0}'.format(max_probable_segment))
        return {0: [max_probable_segment, gps_track[0][0]]}
    elif len(gps_track) == 0:
        raise ValueError('No path')
    # except ValueError as e:
    #     if str(e) == 'Point density error.':
    #         print('Gotcha.')
    #         return e
    #     else:
    #         print(e)
    #         sys.exit(1)


def index_track_points(track, grid):
    """
    Creates a dictionary storing information about each point for the given vehicle. Consecutive points that either a)
    map to the same gridkey in the index grid or b) have the same matching single street network segment candidate are
    ignored in the index to improve processing time and remove unnecessary redundancies from the script's outputs.

    Parameters:
        track (Feature Layer): The in memory feature layer containing the gps points for the current VIN
        grid (dictionary): The point index containing all possible gps point locations and the street network candidate
            segments for each point location

    Output: track_points is a dictionary containing information about the order of points, time of points, and street
        network segment candidates for each point. Key=point order, Value=list including point time (integer) and
        candidate segments with probabilities (dictionary).
    """
    # Create dictionary to store indexed points to run through Viterbi algorithm
    track_points = {}
    # Get count of total amount of points in the track to check for indexing
    track_total = len(track)
    # Initialize point and index counts
    track_count = 0
    point_index = 0
    # Initialize variables to store information about points and previous points, used for determining if a point should
    # be indexed
    candidates = None
    previous_candidates = [None, {}]
    previous_previous_candidates = [None, {}]
    candidate_trigger = False
    previous_gridkey = (None, [None])
    previous_previous_gridkey = (None, [None])
    gridkey_trigger = False
    # track_point_cursor = arcpy.da.SearchCursor(in_table=track, field_names=['latitude', 'longitude', 'fixtimeutf'])
    # For each point in a vehicle's track
    for track_point in track:
        track_count += 1
        # Build the search string for looking up a point's street segment candidates in the index grid
        point_search = build_search_string(track_point[1], track_point[2])
        # Determine if the point exists in the index grid and retrieve street segment candidates
        try:
            candidates = literal_eval(grid[point_search])
        # If the point does not exist in the grid, ignore and move on to the next point
        except KeyError:
            print('{0} does not exist in the gridkey index'.format(point_search))
            continue
        except:
            error_handler('New error, please debug - 0.')
        # If the point is not the final point in the track
        if track_count != track_total:
            # If the point has no street segment candidates, ignore and move on to the next point
            if len(candidates) == 0:
                continue
            # If the point has only one street segment candidate
            if len(candidates) == 1:
                # Check to see if a previous segment with multiple candidates has been stored in memory
                if gridkey_trigger:
                    # If so, index the previous segment and reset the memory trigger
                    track_points[point_index] = previous_gridkey[1]
                    point_index += 1
                    gridkey_trigger = False
                # Make sure the previous gridkey variables are empty
                previous_gridkey = (None, [None])
                previous_previous_gridkey = (None, [None])
                # Check to see if the current point's single candidate is the same as the previous's
                if candidates.keys() == previous_candidates[1].keys():
                    # If so, this is the third or more consecutive point with a single, and matching, street segment
                    # candidate
                    if previous_candidates[1].keys() == previous_previous_candidates[1].keys():
                        # Prepare variables for next point
                        previous_previous_candidates = previous_candidates
                        previous_candidates = [track_point[0], candidates]
                        # Set candidate trigger to true, signifying a single candidate point has been stored in memory
                        candidate_trigger = True
                    # Else this is the second consecutive point with a single, and matching, street segment candidate
                    else:
                        # Prepare variables for next point
                        previous_previous_candidates = previous_candidates
                        previous_candidates = [track_point[0], candidates]
                        # Set candidate trigger to true, signifying a single candidate point has been stored in memory
                        candidate_trigger = True
                # Else if the current point's single candidate is not the same as the previous's
                else:
                    # Check to see if a previous single point candidate has been stored in memory
                    if candidate_trigger:
                        # If so, index the stored point
                        track_points[point_index] = previous_candidates
                        point_index += 1
                        # Prepare variables for next point
                        previous_previous_candidates = [None, {}]
                        previous_candidates = [track_point[0], candidates]
                        # Index the current point
                        track_points[point_index] = [track_point[0], candidates]
                        point_index += 1
                        # Reset the memory trigger
                        candidate_trigger = False
                    # If there is no point stored in memory
                    else:
                        # Prepare variables for next point
                        previous_previous_candidates = [None, {}]
                        previous_candidates = [track_point[0], candidates]
                        # Index the current point
                        track_points[point_index] = [track_point[0], candidates]
                        point_index += 1
            else:
                if candidate_trigger:
                    track_points[point_index] = previous_candidates
                    point_index += 1
                    candidate_trigger = False
                previous_candidates = [None, {}]
                previous_previous_candidates = [None, {}]
                if point_search == previous_gridkey[0]:
                    if previous_gridkey[0] == previous_previous_gridkey[0]:
                        previous_previous_gridkey = previous_gridkey
                        previous_gridkey = (point_search, [track_point[0], candidates])
                        gridkey_trigger = True
                    else:
                        previous_previous_gridkey = previous_gridkey
                        previous_gridkey = (point_search, [track_point[0], candidates])
                        gridkey_trigger = True
                else:
                    if gridkey_trigger:
                        track_points[point_index] = previous_gridkey[1]
                        point_index += 1
                        previous_previous_gridkey = (None, [None])
                        previous_gridkey = (point_search, [track_point[0], candidates])
                        track_points[point_index] = [track_point[0], candidates]
                        point_index += 1
                        gridkey_trigger = False
                    else:
                        previous_previous_gridkey = (None, [None])
                        previous_gridkey = (point_search, [track_point[0], candidates])
                        track_points[point_index] = [track_point[0], candidates]
                        point_index += 1
        # If the point is the final point in the track
        else:
            if candidate_trigger:
                if candidates.keys() != previous_candidates[1].keys():
                    track_points[point_index] = previous_candidates
                    point_index += 1
            if gridkey_trigger:
                if point_search != previous_gridkey:
                    track_points[point_index] = previous_gridkey[1]
                    point_index += 1
            if len(candidates) > 0:
                track_points[point_index] = [track_point[0], candidates]
    # del track_point_cursor
    del candidates
    del previous_candidates
    del previous_previous_candidates
    return track_points


def build_search_string(latitude, longitude):
    """
    Converts latitude and longitude into the same format as the grid key in the grid point index (10 character string).
        A point with latitude = 40.12345 and longitude = -75.12345 will be converted to '0123451234'. This methodology
        only works in an area with an extent covering less than 10 decimal degrees latitude and less than 10 decimal
        degrees longitude. Note that the point index grid rounds to the 4th decimal place, the gridkey built here takes
        this into account.

    Parameters:
        latitude: Latitude of input point to the 5th decimal place (float) -> 40.12345
        longitude: Longitude of input point to the 5th decimal place (float -> -75.12345

    Output: Ten character string representing the grid key for the given point location.
    """

    if len(str(latitude)) >= 8:
        first_five = str(latitude)[1] + str(latitude)[3:7]
    elif len(str(latitude)) < 8:
        first_five = str(latitude)[1] + str(latitude)[3:]
    else:
        first_five = str(latitude)[1] + str(latitude)[3:6]
    while len(first_five) < 5:
        first_five = first_five + '0'

    if len(str(longitude)) >= 9:
        last_five = str(longitude)[2] + str(longitude)[4:8]
    elif len(str(longitude)) < 9:
        last_five = str(longitude)[2] + str(longitude)[4:]
    else:
        last_five = str(longitude)[2] + str(longitude)[4:7]
    while len(last_five) < 5:
        last_five = last_five + '0'

    return first_five + last_five


def calculate_network_transition_probability(segment_1, segment_2, input_graph, end_nodes, net_decay_constant):
    """
        Calculates the transition probability of a vehicle's path going from one street network segment to another,
            based on the network distance between the segments and the resulting path.

        Parameters:
             segment_1: Fist input street network segment
             segment_2: Second input street network segment
             input_graph: networkx graph of the street network segments
             end_nodes: End points of the input street network segments
             net_decay_constant: Distance in meters after which the match probability falls under 0.34 (exponential
                decay), parameter depends on the intervals between successive points in the gps point track

        Output: Returns the transition probability value
        """
    sub_path = []
    segment_2_point = None

    if segment_1 == segment_2:
        # if previous segment candidate is the same as the current segment candidate, distance is 0
        distance = 0
    else:
        # Obtain edges (tuples of endpoints) for segment identifiers
        segment_1_edge = end_nodes[segment_1]
        segment_2_edge = end_nodes[segment_2]

        # Determine segment endpoints of the two segments that are closest to each other
        minimum_pair = [0, 0, 100000]
        for i in range(0, 2):
            for j in range(0, 2):
                d = round(calculate_point_distance(segment_1_edge=segment_1_edge[i],
                                                   segment_2_edge=segment_2_edge[j]), 2)
                if d < minimum_pair[2]:
                    minimum_pair = [i, j, d]
        segment_1_point = segment_1_edge[minimum_pair[0]]
        segment_2_point = segment_2_edge[minimum_pair[1]]

        if segment_1_point == segment_2_point:
            distance = 5
        else:
            try:
                if input_graph.has_node(segment_1_point) and input_graph.has_node(segment_2_point):
                    distance = nx.shortest_path_length(input_graph, segment_1_point, segment_2_point, weight='length')
                    path = nx.shortest_path(input_graph, segment_1_point, segment_2_point, weight='length')
                    path_edges = zip(path, path[1:])
                    sub_path = []
                    for path_edge in path_edges:
                        segment_id = input_graph[path_edge[0]][path_edge[1]]['SEG_ID']
                        sub_path.append(segment_id)
                else:
                    # Node not found in segment's input graph, set distance to a larger number
                    distance = 3 * net_decay_constant
            except nx.NetworkXNoPath:
                # If NetworkX returns a no path error, set  distance to a larger number
                distance = 3 * net_decay_constant
    return calculate_network_distance_probability(distance=distance, net_decay_constant=net_decay_constant), \
        sub_path, segment_2_point


def calculate_point_distance(segment_1_edge, segment_2_edge):
    """
    Calculate the Euclidean distance between the endpoints of two segments

    Parameters:
    segment_1_edge (tuple): The coordinates of the endpoint of the first segment
    segment_2_edge (tuple): The coordinates of the endpoint of the second segment

    Output: point_distance is the distance between the endpoints of the two segments.
    """
    point_distance = sqrt((segment_1_edge[0] - segment_2_edge[0]) ** 2 + (segment_1_edge[1] - segment_2_edge[1]) ** 2)
    return point_distance


def calculate_network_distance_probability(distance, net_decay_constant):
    """
    Calculates the network distance probability that a segment follows another in the vehicle's optimized path.

    Parameters:
         distance: The length of the shortest path between points on two segments
         net_decay_constant: Distance in meters after which the match probability falls under 0.34 (exponential decay),
            parameter depends on the intervals between successive points in the gps point track

    Output: network_distance_probability is the probability value that a segment follows another in the vehicle's
        optimized path.
    """
    distance = float(distance)
    try:
        network_distance_probability = 1 if distance == 0 else round(1 / exp(distance / net_decay_constant), 2)
    except:
        network_distance_probability = round(1 / float('inf'), 2)
    return network_distance_probability


def optimize_solved_path(input_path):
    """
    Removes redundant records from the solved path before writing to the output.

    Parameters:
        input_path (dictionary): The solved network path for the vehicle

    Output: optimized_path (dictionary): Optimized solved network path with redundant records removed
    """

    new_index = 0
    optimized_path = {}
    previous_segment = None

    for index in range(len(input_path) - 1, -1, -1):
        if index != 0:
            if not previous_segment or previous_segment[0] == input_path[index][0]:
                previous_segment = input_path[index]
            else:
                optimized_path[new_index] = previous_segment
                new_index += 1
                previous_segment = input_path[index]
        else:
            if previous_segment[0] == input_path[index][0]:
                optimized_path[new_index] = input_path[index]
            else:
                optimized_path[new_index] = previous_segment
                new_index += 1
                optimized_path[new_index] = input_path[index]
    # TODO Make the following (and above) section of code an option through the config, allow user to choose first and
    # {continued} last point or just last point
    # new_index = 0
    # optimized_path = {}
    # previous_segment = [None, None]
    # previous_previous_segment = [None, None]
    # index_trigger = False
    #
    # for index in range(len(input_path) - 1, -1, -1):
    #     if index != 0:
    #         if previous_segment[0] == input_path[index][0]:
    #             if previous_segment[0] == previous_previous_segment[0]:
    #                 previous_previous_segment = previous_segment
    #                 previous_segment = input_path[index]
    #                 index_trigger = True
    #             else:
    #                 previous_previous_segment = previous_segment
    #                 previous_segment = input_path[index]
    #                 index_trigger = True
    #         else:
    #             if index_trigger:
    #                 optimized_path[new_index] = previous_segment
    #                 new_index += 1
    #                 previous_previous_segment = [None, None]
    #                 previous_segment = input_path[index]
    #                 optimized_path[new_index] = input_path[index]
    #                 new_index += 1
    #                 index_trigger = False
    #             else:
    #                 previous_previous_segment = [None, None]
    #                 previous_segment = input_path[index]
    #                 optimized_path[new_index] = input_path[index]
    #                 new_index += 1
    #     else:
    #         if index_trigger:
    #             if previous_segment[0] != input_path[0][0]:
    #                 optimized_path[new_index] = previous_segment
    #                 new_index += 1
    #         optimized_path[new_index] = input_path[index]
    return optimized_path


def densifier(sparse_points, threshold_meters=120, max_midpoints=5):
    """
    If a vehicle's path cannot be solved, creates additional points along a vehicles route to provide additional
    reference points for route solving. This option can be toggled on / off and the distance threshold can be altered
    in the config file.

    Parameters:
        sparse_points: The input points layer that could not be solved.
        threshold_meters: The maximum distance in meters between points before additional interpolated points and times
            are added to the vehicles path.
        max_midpoints:

    Output: Returns an in memory point layer with additional interpolated point locations and times to assist in solving
        a vehicle's path.
    """
    unsorted_dense_points = []
    previous_sparse_point = None
    threshold_meters = float(threshold_meters)
    for sparse_point in sparse_points:
        unsorted_dense_points.append(sparse_point)
        current_sparse_point = sparse_point
        if previous_sparse_point:
            longitude_1, latitude_1, longitude_2, latitude_2 = map(radians, [previous_sparse_point[2],
                                                                             previous_sparse_point[1],
                                                                             current_sparse_point[2],
                                                                             current_sparse_point[1]])
            distance_longitude = longitude_2 - longitude_1
            distance_latitude = latitude_2 - latitude_1
            total_distance = 6371000 * 2 * asin(sqrt(sin(distance_latitude / 2) ** 2 + cos(latitude_1) * cos(latitude_2)
                                                     * sin(distance_longitude / 2) ** 2))
            if total_distance <= threshold_meters:
                previous_sparse_point = current_sparse_point
                continue
            else:
                length = numpy.array([total_distance])
                bin_list = []
                for i in range(1, max_midpoints + 1):
                    bin_list.append(threshold_meters * i)
                bins = numpy.array(bin_list)
                midpoint_count = int(numpy.digitize(length, bins).item())
                if midpoint_count == max_midpoints:
                    print('Long segment warning.')
                x_delta = (current_sparse_point[2] - previous_sparse_point[2]) / (midpoint_count + 1)
                y_delta = (current_sparse_point[1] - previous_sparse_point[1]) / (midpoint_count + 1)
                time_delta = int((current_sparse_point[0] - previous_sparse_point[0]) / (midpoint_count + 1))
                for i in range(1, midpoint_count + 1):
                    midpoint = (previous_sparse_point[0] + (i * time_delta), previous_sparse_point[1] + (i * y_delta),
                                previous_sparse_point[2] + (i * x_delta))
                    unsorted_dense_points.append(midpoint)
        previous_sparse_point = current_sparse_point
    print(unsorted_dense_points)
    sorted_dense_points = sorted(unsorted_dense_points, key=lambda tup: tup[0])

    return sorted_dense_points


def output_writer(connection, cursor, sql, vin, assignment, match_route, dict_duplicate_check):
    """
    Writes street network segment visits to the output table.

    Parameters:
        connection:
        cursor:
        sql:
        vin: The current vehicle's VIN
        assignment: Vehicle's assignment (Sanitation, Highways, etc)
        match_route: Dictionary containing the records that should be added to the output table

    Output: No output, records are added to pre-existing table
    """
    duplicate_check = True
    # timezone_time_now = dt.now(pytz.timezone(config['inputs']['timezone']))
    # time_difference = int(timezone_time_now.utcoffset().total_seconds())

    # Check to see if the vehicle exists in the output segment visits table
    if vin in dict_duplicate_check:
        # log.info('VIN: {0}'.format(vin))
        # If the vehicle exists, for each new segment visit in the vehicle's solved route
        for index in match_route.values():
            timestamp = dt.utcfromtimestamp(index[1])
            # timestamp = dt.utcfromtimestamp(index[1] + time_difference)
            # Check for script period overlap and handle any overlapping segment visits
            if duplicate_check:
                # If the new segment visit is older than the last segment visit written in the table
                if index[1] < dict_duplicate_check[vin][1]:
                    # If the new segment visit is older than the last segment visit written in the table and the new
                    # visit is at a different segment than the last visit, assume the new record is redundant to
                    # information already in the table and continue
                    if index[0] != dict_duplicate_check[vin][0]:
                        continue
                    # If the segment visit is older than the last segment visit written in the table and the new visit
                    # is at the same segment as the last visit written, assume that there has been a correction made to
                    # the last visit written in the new run of the script.  Adjust the the timestamp to match that of
                    # the new visit.  The original time stamp will possibly reappear in the table with the next segment
                    # written or disappear.  Stop checking for script period overlap.
                    else:
                        # log.info("Changed oid {2} time visited from {0} to {1}".format(dict_duplicate_check[vin][1],
                        # index[1], dict_duplicate_check[vin][2]))
                        segment_visits_update_sql = "UPDATE {0} SET time_visited = '{1}', time_visited_unix = {2} " \
                                                    "WHERE objectid = " \
                                                    "{3}".format(config['outputs']['segment_visits_table'],
                                                                 timestamp, index[1], dict_duplicate_check[vin][2])
                        cursor.execute(segment_visits_update_sql)
                        connection.commit()
                        duplicate_check = False
                # If the new segment visit is at the same time as the last segment visit written in the table
                elif index[1] == dict_duplicate_check[vin][1]:
                    # If the new segment visit is at the same street segment as the last segment visit written in the
                    # table, stop checking for script period overlap, skip this segment, and continue to the next
                    if index[0] == dict_duplicate_check[vin][0]:
                        duplicate_check = False
                        continue
                    # If the new segment visit is at a different street segment as the last segment visit written in the
                    # table, update the record written in the table to reflect the new segment. It is assumed that new
                    # information provided has allowed this record to be solved more accurately than during the previous
                    # script period. Also, stop checking for script period overlap.
                    else:
                        # log.info("Changed oid {2} segment from {0} to {1}".format(dict_duplicate_check[vin][0],
                        #                                                           index[0],
                        #                                                           dict_duplicate_check[vin][2]))
                        segment_visits_update_sql = "UPDATE {0} SET segment_id = '{1}' WHERE objectid = " \
                                                    "{2}".format(config['outputs']['segment_visits_table'],
                                                                 index[0], dict_duplicate_check[vin][2])
                        cursor.execute(segment_visits_update_sql)
                        connection.commit()
                        duplicate_check = False
                # Else the new segment visit is newer than the last segment visit written in the table
                else:
                    # If the new segment visit is at the same street segment as the last segment visit written in the
                    # table, stop checking for script period overlap and update the record written in the table to
                    # reflect the new time visited.  It is assumed that the vehicle had not yet left the street segment
                    # in the previous script period and new information provided has allowed us to more accurately say
                    # when the vehicle last visited the street segment. Also, stop checking for script period overlap.
                    if index[0] == dict_duplicate_check[vin][0]:
                        # log.info("Updated oid {2} time visited from {0} to {1}".format(dict_duplicate_check[vin][1],
                        #                                                                index[1],
                        #                                                                dict_duplicate_check[vin][2]))
                        segment_visits_update_sql = "UPDATE {0} SET time_visited = '{1}', time_visited_unix = {2} " \
                                                    "WHERE objectid = " \
                                                    "{3}".format(config['outputs']['segment_visits_table'],
                                                                 timestamp, index[1], dict_duplicate_check[vin][2])
                        cursor.execute(segment_visits_update_sql)
                        connection.commit()
                        duplicate_check = False
                    # If the new segment visit is at a different street segment than the last segment visit written in
                    # the table, insert the new record and stop checking for script period overlap.
                    else:
                        cursor.execute(sql, {'seg': index[0], 'ts': timestamp, 'ts_unix': index[1], 'vin': vin,
                                             'asg': assignment})
                        connection.commit()
                        duplicate_check = False
            # Script period overlap has already been detected and overlapping segment visits have been handled, insert
            # all new records to the output table
            else:
                cursor.execute(sql,
                               {'seg': index[0], 'ts': timestamp, 'ts_unix': index[1], 'vin': vin,
                                'asg': assignment})
                connection.commit()
    # If the vehicle does not exist, there is no possible script period overlap. Insert all new records to the table.
    else:
        for index in match_route.values():
            timestamp = dt.utcfromtimestamp(index[1])
            # timestamp = dt.utcfromtimestamp(index[1] + time_difference)
            cursor.execute(sql, {'seg': index[0], 'ts': timestamp, 'ts_unix': index[1], 'vin': vin,
                                 'asg': assignment})
            connection.commit()


if __name__ == '__main__':
    # Locate and read config file
    scriptDirectory = os.path.dirname(__file__)
    config = ConfigParser()
    config.read(os.path.join(scriptDirectory, 'config.cfg'))
    # Set up logging
    try:
        log_file_path = os.path.join(scriptDirectory, config['logging']['log_file'])
        log = logging.getLogger('street-segment-identifier')
        log.setLevel(logging.INFO)
        handler = logging.FileHandler(log_file_path)
        handler.setLevel(logging.INFO)
        handlerFormatter = logging.Formatter('%(name)s (%(levelname)s)-%(asctime)s: %(message)s',
                                             '%m/%d/%Y  %I:%M:%S %p')
        handler.setFormatter(handlerFormatter)
        log.addHandler(handler)
    except KeyError:
        error_handler('Make sure config file exists and contains a log_file variable.')

    # timezone_time_now = dt.now(pytz.timezone(config['inputs']['timezone']))
    # time_difference = int(dt.now(pytz.timezone(config['inputs']['timezone'])).utcoffset().total_seconds())
    update_time = int(config['inputs']['update_time_minutes']) * 60
    scriptStart = int(dt.timestamp(dt.now()) // update_time * update_time)
    log.info('Script started using UNIX time: {0} ({1} local time)'.format(
        scriptStart, dt.utcfromtimestamp(scriptStart + int(dt.now(
            pytz.timezone(config['inputs']['timezone'])).utcoffset().total_seconds())).strftime('%c %Z')))
    print('Script started using UNIX time: {0} ({1} local time)'.format(
        scriptStart, dt.utcfromtimestamp(scriptStart + int(dt.now(
            pytz.timezone(config['inputs']['timezone'])).utcoffset().total_seconds())).strftime('%c %Z')))
    print(scriptStart)
    # Determine the beginning of the time period in which points should be pulled from, default is 960 seconds (16
    # minutes) before the current time
    slice_time_start = scriptStart - int(config['inputs']['point_service_pull_time'])
    print(slice_time_start)
    # Prepare inputs for script
    arcpy.env.overwriteOutput = config['environment']['env_overwrite']
    arcpy.env.workspace = config['environment']['env_workspace']
    street_network = config['inputs']['street_network']
    point_grid = os.path.join(arcpy.env.workspace, config['inputs']['point_grid'])
    production_database_dns = "dbname='{0}' user='{1}' host='{2}' password='{3}'".format(
        config['outputs']['database_name'], config['outputs']['database_user'], config['outputs']['database_host'],
        config['outputs']['database_password'])
    production_database_connection = psycopg2.connect(production_database_dns)
    production_database_cursor = production_database_connection.cursor()
    gps_points_select_sql = "SELECT vin, fixtimeutf, latitude, longitude FROM networkfleet_gps " \
                            "WHERE fixtimeutf > {0} AND fixtimeutf < {1} ORDER BY vin asc, fixtimeutf " \
                            "asc".format(slice_time_start, scriptStart)
    segment_visits_select_sql = "SELECT objectid, segment_id, time_visited_unix, sv1.vin FROM {0} sv1 JOIN (SELECT " \
                                "vin, max(time_visited_unix) as maxtime FROM {0} GROUP BY VIN) sv2 ON " \
                                "sv1.vin = sv2.vin AND sv1.time_visited_unix = " \
                                "sv2.maxtime".format(config['outputs']['segment_visits_table'])
    vehicle_assignment_select_sql = "SELECT vin, responsibility, snow_yn FROM networkfleet_vehicles_streets"
    segment_visits_insert_sql = "INSERT INTO {0} (segment_id, time_visited, time_visited_unix, vin, " \
                               "vehicle_assignment) VALUES (%(seg)s, %(ts)s, %(ts_unix)s, %(vin)s, " \
                               "%(asg)s)".format(config['outputs']['segment_visits_table'])
    most_recent_visit_update_sql = "UPDATE {0} SET vin = a.vin, time_visited=a.time_visited, time_visited_unix = " \
                                   "a.time_visited_unix, vehicle_assignment=a.vehicle_assignment, seconds_since_visit" \
                                   " = {2} - a.time_visited_unix, time_since_visit = CASE WHEN trim(leading ' ' from" \
                                   " to_char(FLOOR(({2} - a.time_visited_unix) / 86400), '0009')) = '0000' THEN TRIM(" \
                                   " LEADING ' ' from TO_CHAR(FLOOR((({2} - a.time_visited_unix) / 3600) - FLOOR(({2}" \
                                   " - a.time_visited_unix) / 86400) * 24), '09')) || ':' || TRIM( LEADING ' ' from" \
                                   " TO_CHAR(FLOOR((({2} - a.time_visited_unix) / 60) - FLOOR(({2} - " \
                                   "a.time_visited_unix) / 3600) * 60), '09'))  || ':' || TRIM( LEADING ' ' from " \
                                   "TO_CHAR(({2} - a.time_visited_unix) - FLOOR(({2} - a.time_visited_unix) / 60) * " \
                                   "60, '09')) WHEN trim(leading ' ' from to_char(FLOOR(({2} - a.time_visited_unix) /" \
                                   " 86400), '0009')) = '0001' THEN '1 day, ' || TRIM( LEADING ' ' from TO_CHAR(FLOOR" \
                                   "((({2} - a.time_visited_unix) / 3600) - FLOOR(({2} - a.time_visited_unix) / " \
                                   "86400) * 24), '09')) || ':' || TRIM( LEADING ' ' from TO_CHAR(FLOOR((({2} - " \
                                   "a.time_visited_unix) / 60) - FLOOR(({2} - a.time_visited_unix) / 3600) * 60), " \
                                   "'09'))  || ':' || TRIM( LEADING ' ' from TO_CHAR(({2} - a.time_visited_unix) - " \
                                   "FLOOR(({2} - a.time_visited_unix) / 60) * 60, '09')) ELSE trim(leading '0' from " \
                                   "trim(leading ' ' from to_char(FLOOR(({2} - a.time_visited_unix) / 86400), '0009'" \
                                   "))) || ' days, ' || TRIM( LEADING ' ' from TO_CHAR(FLOOR((({2} - " \
                                   "a.time_visited_unix) / 3600) - FLOOR(({2} - a.time_visited_unix) / 86400) * 24)," \
                                   " '09')) || ':' || TRIM( LEADING ' ' from TO_CHAR(FLOOR((({2} - " \
                                   "a.time_visited_unix) / 60) - FLOOR(({2} - a.time_visited_unix) / 3600) * 60), " \
                                   "'09'))  || ':' || TRIM( LEADING ' ' from TO_CHAR(({2} - a.time_visited_unix) - " \
                                   "FLOOR(({2} - a.time_visited_unix) / 60) * 60, '09')) END FROM {1} a INNER JOIN(" \
                                   "SELECT segment_id, MAX(time_visited_unix) as time_visited_unix FROM {1} GROUP BY " \
                                   "segment_id) b ON a.segment_id = b.segment_id AND a.time_visited_unix = " \
                                   "b.time_visited_unix WHERE {0}.segment_id = a.segment_id".format(
                                                        config['outputs']['most_recent_visit_table'],
                                                        config['outputs']['segment_visits_table'], scriptStart)
    rubbish_visit_update_sql = "UPDATE {0} SET rubbish_vin = a.vin, rubbish_time_visited=a.time_visited, " \
                               "rubbish_time_visited_unix = a.time_visited_unix FROM {1} a INNER JOIN(SELECT " \
                               "segment_id, MAX(time_visited_unix) as time_visited_unix FROM {1} GROUP BY segment_id)" \
                               " b ON a.segment_id = b.segment_id AND a.time_visited_unix = b.time_visited_unix " \
                               "WHERE {0}.segment_id = a.segment_id AND a.vehicle_assignment = 'RUBBISH' AND " \
                               "to_timestamp(a.time_visited_unix) >= now()::date".format(
                                        config['outputs']['sanitation_visit_table'],
                                        config['outputs']['segment_visits_table'],
                                        scriptStart)
    recycling_visit_update_sql = "UPDATE {0} SET recycling_vin = a.vin, recycling_time_visited=a.time_visited, " \
                                 "recycling_time_visited_unix = a.time_visited_unix FROM {1} a INNER JOIN(SELECT " \
                                 "segment_id, MAX(time_visited_unix) as time_visited_unix FROM {1} GROUP BY " \
                                 "segment_id) b ON a.segment_id = b.segment_id AND a.time_visited_unix = " \
                                 "b.time_visited_unix WHERE {0}.segment_id = a.segment_id AND a.vehicle_assignment " \
                                 "= 'RECYCLE' AND to_timestamp(a.time_visited_unix) >= now()::date".format(
                                        config['outputs']['sanitation_visit_table'],
                                        config['outputs']['segment_visits_table'],
                                        scriptStart)
    sanitation_visit_status_update_sql = "UPDATE {0} AS svs SET visited_status = CASE WHEN rubbish_vin IS NOT NULL " \
                                         "AND recycling_vin IS NOT NULL THEN 'BOTH' WHEN rubbish_vin IS NOT NULL " \
                                         "THEN 'RUBBISH' WHEN recycling_vin IS NOT NULL THEN 'RECYCLE' ELSE " \
                                         "NULL END".format(config['outputs']['sanitation_visit_table'])

    points = point_retriever(cursor=production_database_cursor, sql=gps_points_select_sql)

    # Read / index street network and create network graph
    segment_info = index_network_segment_info(input_network=street_network)
    endpoints = segment_info[0]
    lengths = segment_info[1]
    network_graph = create_network_graph(input_network=street_network, segment_lengths=lengths)

    # Read in the point grid to capture candidate street network segments for each possible gps point
    grid_index = read_point_grid(point_grid)

    # Read in vehicle assignments for streets vehicles
    production_database_cursor.execute(vehicle_assignment_select_sql)
    dict_vehicle_assignment = {}
    for vin in production_database_cursor.fetchall():
        # print(vin)
        dict_vehicle_assignment[vin[0]] = [vin[1], vin[2]]
    # for item in dict_vehicle_assignment.items():
    #     print(item)
    # Read in most recent visits for each segment
    production_database_cursor.execute(segment_visits_select_sql)
    dict_most_recent_vin_record = {}
    for vin in production_database_cursor.fetchall():
        dict_most_recent_vin_record[vin[3]] = [vin[1], vin[2], vin[0]]
    # for k, v in dict_most_recent_vin_record.items():
    #     print(k, v)

    vin_points = None
    for vin_number in points:  # TODO adjust from here
        if vin_number in dict_vehicle_assignment.keys():
            # print(dict_vehicle_assignment[vin_number][0])
            try:
                print('Processing {0}'.format(vin_number))
                vin_points = points[vin_number]
                mapped_path = street_seg_identifier(gps_points=vin_points, grid=grid_index,
                                                    net_decay_constant=config['inputs']['net_decay_constant'])
                output_writer(connection=production_database_connection, cursor=production_database_cursor,
                              sql=segment_visits_insert_sql, vin=vin_number,
                              assignment=dict_vehicle_assignment[vin_number][0], match_route=mapped_path,
                              dict_duplicate_check=dict_most_recent_vin_record)
                del mapped_path
            except ValueError as e:
                if str(e) == 'Point density error.':
                    if config['options']['densify']:
                        try:
                            print('Attempting to densify points and repeat the identifier function.')
                            dense_points = densifier(vin_points,
                                                     threshold_meters=config['options']['densify_threshold'])
                            mapped_path = street_seg_identifier(gps_points=dense_points, grid=grid_index,
                                                                net_decay_constant=
                                                                config['inputs']['net_decay_constant'])
                            output_writer(connection=production_database_connection, cursor=production_database_cursor,
                                          sql=segment_visits_insert_sql, vin=vin_number,
                                          assignment=dict_vehicle_assignment[vin_number][0], match_route=mapped_path,
                                          dict_duplicate_check=dict_most_recent_vin_record)
                            del dense_points
                            del mapped_path
                        except ValueError as e:
                            if str(e) == 'Point density error.':
                                print('Vehicle path could not be solved for {0}.'.format(vin_number))
                                continue
                            else:
                                error_handler('New error, please debug - 1.')
                    else:
                        print('Vehicle path could not be solved, increase decay or select densify option in config.')
                elif str(e) == 'No path':
                    print('No path for vin {0}'.format(vin_number))
                    continue
                else:
                    error_handler('New error, please debug - 2.')
            except:
                error_handler('New error, please debug - 3.')
        else:
            continue
    # print(most_recent_visit_update_sql)
    production_database_cursor.execute(most_recent_visit_update_sql)
    production_database_connection.commit()
    production_database_cursor.execute(rubbish_visit_update_sql)
    production_database_connection.commit()
    production_database_cursor.execute(recycling_visit_update_sql)
    production_database_connection.commit()
    production_database_cursor.execute(sanitation_visit_status_update_sql)
    production_database_connection.commit()
    production_database_cursor.close()
    production_database_connection.close()
    log.info('Script ended: {0}'.format(dt.now().strftime('%c %Z')))
    print('Script ended: {0}'.format(dt.now().strftime('%c %Z')))

# TODO Look at flags
# TODO For vehicles with points outside of the grid (outside of the city), check if any points exist in the city, if so:
    # TODO create path from points inside the city only - This should already be occurring
    # TODO Split path into multiple if the vehicle re-enters the city
