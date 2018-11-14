"""
Script: Street Segment Identifier
File Name: street-segment-identifier.py
Version: 1.0
Date Created: 10/24/2018
Author: Tim Haynes & Paul Sesink Clee
Last Update: 11/6/2018
Updater: Tim Haynes

Summary: Script for consuming NetworkFleet GPS points and adapting them for street segment outputs.

Credit: This script was adapted from 'mapmatching' simonscheider (https://github.com/simonscheider/mapmatching)
"""

# region Import libraries
import os
import sys
try:
    from ast import literal_eval
    # from collections import OrderedDict
    from configparser import ConfigParser
    import datetime
    # from datetime import date, timedelta, timezone
    # from itertools import repeat
    import arcpy
    import logging
    from math import exp, sqrt
    import networkx as nx
    import numpy
    import traceback
    # import time
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


def point_retriever():
    """Retrieves points from hosted feature service containing gps points from GeoEvent Server & Network Fleet for
     previous config defined amount of time. Projects points into meter based projection in one of two ways depending on
     configuration. If default transformation method is acceptable, api_project should be set to True.  Otherwise, it
     must be set to False, with a transformation method provided."""
    current_time_utc = 1541081729  # int(datetime.datetime.timestamp(datetime.datetime.now()))
    start_slice_utc = current_time_utc - int(config['inputs']['point_service_pull_time'])
    baseURL = config['inputs']['point_service_baseurl']
    where = config['inputs']['point_service_where'].format(start_slice_utc)
    fields = config['inputs']['point_service_fields']
    if bool(config['projections']['api_project']):
        query_projection = config['projections']['meter_based_projection']
        query = config['inputs']['point_service_query_project'].format(where, fields, query_projection)
    else:
        query = config['inputs']['point_service_query'].format(where, fields)
    feature_service_url = baseURL + query
    fs = arcpy.FeatureSet()
    fs.load(feature_service_url)
    arcpy.CopyFeatures_management(in_features=fs, out_feature_class='in_memory/mem_IncomingPoints')
    del fs
    if not bool(config['projections']['api_project']):
        arcpy.Project_management(in_dataset='in_memory/mem_IncomingPoints',
                                             out_dataset=config['working']['projected_points'],
                                             out_coor_system=config['projections']['meter_based_projection'],
                                             transform_method=config['projections']['transformation_method'])
        incoming_points = arcpy.CopyFeatures_management(in_features=config['working']['projected_points'],
                                                        out_feature_class='in_memory/mem_IncomingPoints_proj')
    else:
        incoming_points = 'in_memory/mem_IncomingPoints'
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
            segment_cursor = arcpy.da.SearchCursor(in_table=input_network, field_names=['OBJECTID', 'SHAPE@'])
            # TODO OBJECTID needs to be replaced with seg_id in above cursor for production
            segment_end_points = {}
            segment_lengths = {}
            for segment in segment_cursor:
                segment_end_points[segment[0]] = ((segment[1].firstPoint.X, segment[1].firstPoint.Y), (segment[1].lastPoint.X, segment[1].lastPoint.Y))
                segment_lengths[segment[0]] = segment[1].length
            del segment_cursor
            log.info('Number of segments in the input network: {0}'.format(len(segment_end_points)))
            print('Number of segments in the input network: {0}'.format(len(segment_end_points)))
            return segment_end_points, segment_lengths
        else:
            raise FileNotFoundError('Street Network file not found. Make sure it exists and the config file is '
                                    'correct.')
    except FileNotFoundError as e:
        error_handler(str(e))


def create_network_graph(input_network, segment_lengths):
    """
    TODO Write function summary

    Parameters:
        input_network (shapefile projected in meters): Street network gps points should be matched to.
        segment_lengths (dict): Output from index_network_segment_info function. Key=seg_id, Value=length of segment

    Output: TODO Write function output
    """
    try:
        network_file_path = str(os.path.join(arcpy.env.workspace, input_network))
        if os.path.isfile(network_file_path):
            graph = nx.read_shp(network_file_path)
            # Select the largest connected component of the network graph.
            subgraph = list(nx.connected_component_subgraphs(graph.to_undirected()))[0]
            log.info('Graph size (excluding unconnected parts): {0}'.format(len(graph)))
            print('Graph size (excluding unconnected parts): {0}'.format(len(graph)))
            # Append the length of each road segment as an attribute to the edges in the network subgraph.
            for node_0, node_1 in subgraph.edges():
                segment_id = subgraph[node_0][node_1]['OBJECTID']
                # TODO OBJECTID needs to be replaced with seg_id in above line for production
                subgraph[node_0][node_1]['length'] = segment_lengths[segment_id]
            return subgraph
        else:
            raise FileNotFoundError('Street Network file not found. Make sure it exists and the config file is '
                                    'correct.')
    except FileNotFoundError as e:
        error_handler(str(e))


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
                point_grid_index[grid_point[0]] = grid_point [1]
            del grid_cursor
            print('Point grid size: {0}'.format(len(point_grid_index)))
            return point_grid_index
        else:
            raise FileNotFoundError('Point grid file not found. Make sure it exists and the config file is correct.')
    except FileNotFoundError as e:
        error_handler(str(e))


def street_seg_identifier(gps_points, grid, net_decay_constant=30, euclidean_decay_constant=10,
                          max_distance_constant=50):
    """
    TODO Write function summary

    Parameters: TODO define parameters
        gps_points:
        grid:
        net_decay_constant:
        euclidean_decay_constant:
        max_distance_constant:

    Output: TODO Write output summary
    """
    # Make sure constants are floats
    net_decay = float(net_decay_constant)
    euclidean_decay = float(euclidean_decay_constant)
    max_distance = float(max_distance_constant)

    # Initialize the path array to store
    path_array = [{}]

    # Read in points along with time and street network segment candidates for each
    gps_track = index_track_points(gps_points, grid)

    # TODO Write duplicated segments, yards, etc directly into output table, remove from gps track
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
                        sub_path_time -= 1  # TODO make sure this is working in the correct direction, may want to interpolate instead of subtracting one second from next/previous point
                        solved_path[solved_path_index] = [segment, sub_path_time]
                        solved_path_index += 1
                solved_path[solved_path_index] = [previous_state_segment, sub_path_time]
                solved_path_index += 1
                previous_state = previous_state_segment
            except KeyError:
                print('Key Error while finding path.')
                continue

        if len(solved_path) > 1:
            optimized_solved_path = optimize_solved_path(input=solved_path)
        elif len(solved_path) == 1:
            print('How did this happen?')
            pass
        else:
            print('See if this ever happens') # Todo see if this ever happens
        return optimized_solved_path

    # Else if the GPS track has exactly one point, choose the segment candidate that has the highest probability and
    # write it directly to the output table
    elif len(gps_track) == 1:
        print('Found a gps_track with length == 1.')
        max_probable_segment = max(gps_track[0][1], key=lambda key: gps_track[0][1][key])
        print('Max probable segment is {0}'.format(max_probable_segment))
        output_writer(table=production_table, vin=vin_number, match_route={max_probable_segment: gps_track[0][0]})
    # except ValueError as e:
    #     if str(e) == 'Point density error.':
    #         print('Gotcha.')
    #         return e
    #     else:
    #         print(e)
    #         sys.exit(1)


def index_track_points(track, grid):
    """
    Creates a dictionary storing information about each point for the given vehicle. TODO add info about de-duplication

    Parameters:TODO define parameters
        track ():
        grid (dictionary):

    Output: track_points is a dictionary containing information about the order of points, time of points, and street
        network segment candidates for each point. Key=point order, Value=list including point time (integer) and
        candidate segments with probabilities (dictionary).
    """
    # Create dictionary to store indexed points to run through Viterbi algorithm
    track_points = {}
    # Get count of total amount of points in the track to check for indexing
    track_total = arcpy.GetCount_management(track)
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

    track_point_cursor = arcpy.da.SearchCursor(in_table=track, field_names=['latitude', 'longitude', 'fixtimeutf'])
    # For each point in a vehicle's track
    for track_point in track_point_cursor:
        track_count += 1
        # Build the search string for looking up a point's street segment candidates in the index grid
        point_search = build_search_string(track_point[0], track_point[1])
        # Determine if the point exists in the index grid and retrieve street segment candidates
        try:
            candidates = literal_eval(grid[point_search])
        # If the point does not exist in the grid, ignore and move on to the next point
        except KeyError:
            print('{0} does not exist in the gridkey index'.format(point_search))
            continue
        except:
            error_handler('New error, please debug.')
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
                        previous_candidates = [track_point[2], candidates]
                        # Set candidate trigger to true, signifying a single candidate point has been stored in memory
                        candidate_trigger = True
                    # Else this is the second consecutive point with a single, and matching, street segment candidate
                    else:
                        # Prepare variables for next point
                        previous_previous_candidates = previous_candidates
                        previous_candidates = [track_point[2], candidates]
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
                        previous_candidates = [track_point[2], candidates]
                        # Index the current point
                        track_points[point_index] = [track_point[2], candidates]
                        point_index += 1
                        # Reset the memory trigger
                        candidate_trigger = False
                    # If there is no point stored in memory
                    else:
                        # Prepare variables for next point
                        previous_previous_candidates = [None, {}]
                        previous_candidates = [track_point[2], candidates]
                        # Index the current point
                        track_points[point_index] = [track_point[2], candidates]
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
                        previous_gridkey = (point_search, [track_point[2], candidates])
                        gridkey_trigger = True
                    else:
                        previous_previous_gridkey = previous_gridkey
                        previous_gridkey = (point_search, [track_point[2], candidates])
                        gridkey_trigger = True
                else:
                    if gridkey_trigger:
                        track_points[point_index] = previous_gridkey[1]
                        point_index += 1
                        previous_previous_gridkey = (None, [None])
                        previous_gridkey = (point_search, [track_point[2], candidates])
                        track_points[point_index] = [track_point[2], candidates]
                        point_index += 1
                        gridkey_trigger = False
                    else:
                        previous_previous_gridkey = (None, [None])
                        previous_gridkey = (point_search, [track_point[2], candidates])
                        track_points[point_index] = [track_point[2], candidates]
                        point_index += 1
        # If the point is the final point in the track
        else:
            # TODO We've ignored if the point before the final point should be skipped.  Check this logic
            if candidate_trigger:

                track_points[point_index] = previous_candidates
                point_index += 1
            if gridkey_trigger:
                track_points[point_index] = previous_gridkey[1]
                point_index += 1
            if len(candidates) > 0:
                track_points[point_index] = [track_point[2], candidates]
    # Todo delete variables
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

    if len(str(latitude)) == 8:
        first_five = str(latitude)[1] + str(latitude)[3:-1]
    else:
        first_five = str(latitude)[1] + str(latitude)[3:]
    while len(first_five) < 5:
        first_five = first_five + '0'

    if len(str(longitude)) == 9:
        last_five = str(longitude)[2] + str(longitude)[4:-1]
    else:
        last_five = str(longitude)[2] + str(longitude)[4:]
    while len(last_five) < 5:
        last_five = last_five + '0'

    return first_five + last_five


def calculate_network_transition_probability(segment_1, segment_2, input_graph, end_nodes, net_decay_constant):
    """
        TODO Write summary

        Parameters: TODO define parameters
             segment_1:
             segment_2:
             input_graph:
             end_nodes:
             net_decay_constant:

        Output: TODO Write output summary
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
                d = round(calculate_point_distance(segment_1_edge=segment_1_edge[i], segment_2_edge=segment_2_edge[j]), 2)
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
                        segment_id = input_graph[path_edge[0]][path_edge[1]]['OBJECTID']  # TODO change to segment id
                        sub_path.append(segment_id)
                else:
                    # Node not found in segment's input graph, set distance to a larger number
                    distance = 3 * net_decay_constant
            except nx.NetworkXNoPath:
                # If NetworkX returns a no path error, set  distance to a larger number
                distance = 3 * net_decay_constant
    return calculate_network_distance_probability(distance=distance, net_decay_constant=net_decay_constant), sub_path, segment_2_point



def calculate_point_distance(segment_1_edge, segment_2_edge):
    """
    Calculate the Euclidean distance between the endpoints of two segments

    Parameters: TODO define parameters
    segment_1_edge:
    segment_2_edge:

    Output: TODO write output summary
    """
    point_distance = sqrt((segment_1_edge[0] - segment_2_edge[0]) ** 2 + (segment_1_edge[1] - segment_2_edge[1]) ** 2)
    return point_distance


def calculate_network_distance_probability(distance, net_decay_constant):
    """
    TODO Write summary

    Parameters: TODO define parameters
         distance:
         net_decay_constant:

    Output: TODO Write output summary
    """
    distance = float(distance)
    try:
        network_distance_probability = 1 if distance == 0 else round(1 / exp(distance / net_decay_constant), 2)
    except:
        network_distance_probability = round(1 / float('inf'), 2)
    return network_distance_probability


def optimize_solved_path(input):
    """
    TODO Write summary

    Parameters: TODO define parameters
        input:

    Output: TODO Write output summary
    """
    for index in range(len(input), -1, -1):
        pass
        # TODO Adapt code from scratch 4 to fit in this function


def densifier(sparse_points, threshold_meters=120):
    # TODO Fix to set up for gridkey creation
    """
    TODO Write function summary.

    Parameters: TODO define parameters
    input:

    Output: TODO write output summary.
    """
    sparse_points = arcpy.CopyFeatures_management(sparse_points, 'in_memory/sparse_points')
    sparse_cursor = arcpy.da.SearchCursor(in_table=sparse_points, field_names=['SHAPE@XY', 'fixtimeutf'])
    dense_cursor = arcpy.da.InsertCursor(in_table=sparse_points, field_names=['SHAPE@XY', 'fixtimeutf'])
    previous_sparse_point = None
    threshold_meters = float(threshold_meters)
    point_list = []
    for sparse_point in sparse_cursor:
        point_list.append((sparse_point[0][0], sparse_point[0][1], sparse_point[1]))
    del sparse_cursor
    for listed_point in point_list:
        current_sparse_point = (listed_point[0], listed_point[1], listed_point[2])
        if previous_sparse_point:
            # Calculate the distance between the current point and the previous
            distance = sqrt((current_sparse_point[0] - previous_sparse_point[0]) ** 2 + (current_sparse_point[1] - previous_sparse_point[1]) ** 2)
            # If distance is < threshold meters, there is no need to add additional points
            if distance < threshold_meters:
                previous_sparse_point = current_sparse_point
                continue
            # Else calculate the number of necessary midpoints
            d = numpy.array([distance])
            bins = numpy.array([threshold_meters, threshold_meters * 2, threshold_meters * 3, threshold_meters * 4,
                                threshold_meters * 5])
            midpoint_count = int(numpy.digitize(d, bins).item())
            if midpoint_count == 5:
                print('Long segment warning.')
            # Calculate the distances new points will be from previous points on x and y axis
            x_delta = (current_sparse_point[0] - previous_sparse_point[0]) / float(midpoint_count + 1)
            y_delta = (current_sparse_point[1] - previous_sparse_point[1]) / float(midpoint_count + 1)
            time_delta = (current_sparse_point[2] - previous_sparse_point[2]) / float(midpoint_count + 1 )
            # Create new points and adjust the time of each point to maintain order
            for count in range(1, midpoint_count + 1):
                midpoint = (previous_sparse_point[0] + count * x_delta, previous_sparse_point[1] + count * y_delta)
                dense_cursor.insertRow(((midpoint[0], midpoint[1]), previous_sparse_point[2] + count * time_delta))
        previous_sparse_point = current_sparse_point
    del dense_cursor
    return arcpy.Sort_management(in_dataset=sparse_points, out_dataset='in_memory/dense_points', sort_field=[['fixtimeutf', 'ASCENDING']])

def output_writer(table, vin, match_route):
    """
    Writes street network segment visits to the output table.

    Parameters:TODO define parameters
        table:
        vin:
        match_route:

    Output: TODO Write output summary
    """
    fields = ['SEGMENT_ID', 'TIME', 'VIN']
    output_cursor = arcpy.da.InsertCursor(table, fields)
    for seg, timestamp in match_route.items():
        if isinstance(timestamp, list) is True:
            try:
                for item in timestamp:
                    output_cursor.insertRow((str(seg), str(item), str(vin)))
            except RuntimeError:
                print(seg)
                print(type(seg))
                output_cursor.insertRow((str(seg), str(item), str(vin)))
        else:
            output_cursor.insertRow((str(seg), str(timestamp), str(vin)))


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
        handlerFormatter = logging.Formatter('%(name)s (%(levelname)s)-%(asctime)s: %(message)s', '%m/%d/%Y  %I:%M:%S %p')
        handler.setFormatter(handlerFormatter)
        log.addHandler(handler)
        log.info('Script started: {0}'.format(datetime.datetime.now().strftime('%c %Z')))
    except KeyError:
        error_handler('Make sure config file exists and contains a log_file variable under "street-segment-identifier".')

    scriptStart = datetime.datetime.timestamp(datetime.datetime.now())
    print('Script started: {0}'.format(datetime.datetime.now().strftime('%c %Z')))

    # Prepare inputs for script
    arcpy.env.overwriteOutput = config['environment']['env_overwrite']
    arcpy.env.workspace = config['environment']['env_workspace']
    street_network = config['inputs']['street_network']
    points = point_retriever()
    point_grid = os.path.join(arcpy.env.workspace, config['inputs']['point_grid'])
    production_table = os.path.join(arcpy.env.workspace, config['outputs']['production_table'])

    # Read / index street network and create network graph
    segment_info = index_network_segment_info(input_network=street_network)
    endpoints = segment_info[0]
    lengths = segment_info[1]
    network_graph = create_network_graph(input_network=street_network, segment_lengths=lengths)

    # Read in the point grid to capture candidate street network segments for each possible gps point
    grid_index = read_point_grid(point_grid)
    arcpy.Sort_management(in_dataset=points, out_dataset='in_memory/mem_points_sorted',
                          sort_field=[['vin', 'ASCENDING'], ['fixtimeutf', 'ASCENDING']])
    # arcpy.FeatureClassToFeatureClass_conversion(in_features='in_memory/mem_points_sorted', out_path='S:\\Groups\\Enabling Technology Services\\GSG\\Projects\\Open_Projects\\GeoEvent', out_name='testtesttest')
    # arcpy.Copy_management(in_data='in_memory/mem_points_sorted', out_data='S:\\Groups\\Enabling Technology Services\\GSG\\Projects\\Open_Projects\\GeoEvent\\MapMatching.gdb\\testtesttest')
    point_cursor = arcpy.da.SearchCursor(in_table='in_memory/mem_points_sorted', field_names=['vin'])
    current_vin = None
    list_vins = []
    for point in point_cursor:
        if not current_vin or current_vin != point[0]:
            list_vins.append(point[0])
        else:
            continue
        current_vin = point[0]
    for vin_number in list_vins:
        try:
            print('Processing {0}'.format(vin_number))
            arcpy.MakeFeatureLayer_management(in_features='in_memory/mem_points_sorted', out_layer='mem_vin_points',
                                              where_clause=""" "vin" LIKE '{0}' """.format(vin_number),
                                              workspace='in_memory')

            # TODO Test if this works faster by querying insdie of cursor in index_track_points instead of creating feature
            # layer for each vin
            street_seg_identifier(gps_points='mem_vin_points', grid=grid_index, net_decay_constant=30,
                                  euclidean_decay_constant=10, max_distance_constant=50)  # TODO place constants in config
        except ValueError as e:
            # todo try statement to prevent extra densification loops
            if str(e) == 'Point density error.':
                if config['options']['densify']:
                    try:
                        print('Attempting to densify points and repeat the identifier function.')
                        dense_points = densifier('mem_vin_points', threshold_meters=config['options']['densify_threshold'])
                        # arcpy.FeatureClassToFeatureClass_conversion(in_features=dense_points,
                        #                                             out_path='S:\\Groups\\Enabling Technology Services\\GSG\\Projects\\Open_Projects\\GeoEvent',
                        #                                             out_name='testdense12')
                        # break
                        street_seg_identifier(gps_points=dense_points, grid=grid_index, net_decay_constant=30,
                                              euclidean_decay_constant=10, max_distance_constant=50)  # TODO place constants in config
                    except ValueError as e:
                        if str(e) == 'Point density error.':
                            print('Path could not be solved for {0}.'.format(vin_number))
                            continue
                        else:
                            error_handler('New error, please debug.')
                else:
                    print('Vehicle path could not be solved, increase decay or select densify option in config.')
        except:
            error_handler('New error, please debug.')
    print('Script ended: {0}'.format(datetime.datetime.now().strftime('%c %Z')))

# TODO Error handle the bottleneck on point retrieval, including potential dam breaks
#   - Dynamic window, time to last time recorded instead of constant window
#   - If max record count, break into pieces
# TODO Maintain some kind of point id through the seg visit table so records can be compared to the raw?
# TODO Look at flags
# TODO See if VIN and Assignment come through added densify points