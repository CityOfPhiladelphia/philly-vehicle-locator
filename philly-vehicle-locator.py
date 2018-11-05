"""
Script: Street Segment Identifier
File Name: street-segment-identifier.py
Version: 1.0
Date Created: 10/24/2018
Author: Tim Haynes & Paul Sesink Clee
Last Update: 10/26/2018
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
    # import math
    import networkx as nx
    # import numpy
    import traceback
    # import time
except ImportError:
    print('Import Error: Missing one or more libraries.')
    print(traceback.format_exc())
    sys.exit(1)
# endregion


def errorhandler(logstring='Script failed.'):
    """Log errors as critical, including user input log string and traceback information. Print same information in
    console."""
    log.critical(logstring)
    log.critical(traceback.format_exc())
    print(logstring)
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
        errorhandler(str(e))


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
        errorhandler(str(e))


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
        errorhandler(str(e))


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

    if len(gps_track) > 1:
        # Initialize the first point
        for segment_candidate in gps_track[0][1]:
            path_array[0][segment_candidate] = {'probability': gps_track[0][1][segment_candidate],
                                                'previous_segment': None, 'path': [], 'path_nodes': [],
                                                'time': gps_track[0][0]}
        # Run the Viterbi algorithm
        # for
        # TODO Continue writing function
    elif len(gps_track) == 1:
        print('Found a gps_track with length == 1.')
        max_probable_segment = max(gps_track[0][1], key=lambda key: gps_track[0][1][key])
        print('Max probable segment is {0}'.format(max_probable_segment))
        output_writer(table=production_table, vin=vin_number, match_route={max_probable_segment: gps_track[0][0]})
    else:
        pass
        # TODO Do we need to do anything in this scenario


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
    # TODO test the logic in this function to make sure it's working correctly.
    track_points = {}
    track_total = int(arcpy.GetCount_management(track))
    track_count = 0
    point_index = 0
    candidates = None
    previous_candidates = [None, None]
    previous_previous_candidates = [None, None]
    candidate_trigger = False
    previous_gridkey = (None, [None])
    previous_previous_gridkey = (None, [None])
    gridkey_trigger = False

    track_point_cursor = arcpy.da.SearchCursor(in_table=track, field_names=['latitude', 'longitude', 'fixtimeutf'])
    for track_point in track_point_cursor:
        track_count += 1
        if track_count != track_total:
            point_search = build_search_string(track_point[0], track_point[1])
            try:
                candidates = literal_eval(grid[point_search])
            except KeyError:
                print('{0} does not exist in the grid'.format(point_search))
            except:
                errorhandler('New error, please debug.')
            if len(candidates) == 1:
                if gridkey_trigger:
                    track_points[point_index] = previous_gridkey[1]
                    point_index += 1
                    gridkey_trigger = False
                previous_gridkey = (None, [None])
                previous_previous_gridkey = (None, [None])
                if candidates.keys() == previous_candidates[1].keys():
                    if previous_candidates[1].keys() == previous_previous_candidates[1].keys():
                        previous_previous_candidates = previous_candidates
                        previous_candidates = [track_point[2], candidates]
                        candidate_trigger = True
                    else:
                        previous_previous_candidates = previous_candidates
                        previous_candidates = [track_point[2], candidates]
                        candidate_trigger = True
                else:
                    if candidate_trigger:
                        track_points[point_index] = previous_candidates
                        point_index += 1
                        previous_previous_candidates = [None, None]
                        previous_candidates = [track_points[2], candidates]
                        track_points[point_index] = [track_point[2], candidates]
                        point_index += 1
                        candidate_trigger = False
                    else:
                        previous_previous_candidates = [None, None]
                        previous_candidates = [track_point[2], candidates]
                        track_points[point_index] = [track_point[2], candidates]
                        point_index += 1
            else:
                if candidate_trigger:
                    track_points[point_index] = previous_candidates
                    point_index += 1
                    candidate_trigger = False
                previous_candidates = [None, None]
                previous_previous_candidates = [None, None]
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
        else:
            if candidate_trigger:
                track_points[point_index] = previous_candidates
                point_index += 1
            if gridkey_trigger:
                track_points[point_index] = previous_gridkey[1]
                point_index += 1
            track_points[point_index] = [track_point[2], candidates]
    # Todo delete variables
    return track_points


def build_search_string(latitude, longitude):
    """
    Converts latitude and longitude into the same format as the grid key in the grid point index (10 character string).
        A point with latidude = 40.12345 and longitude = -75.12345 will be converted to '0123451234'. This methodology
        only works in an area with an extent covering less than 10 decimal degrees latitude and less than 10 decimal
        degrees longitude. Note that the point index grid rounds to the 4th decimal place, the gridkey built here takes
        this into account.

    Parameters:
        latitude: Latitude of input point to the 5th decimal place (float) -> 40.12345
        longitude: Longitude of input point to the 5th decimal place (float -> -75.12345

    Output: Ten character string representing the grid key for the given point location.
    """
    first_five = str(latitude)[1] + str(latitude)[3:-1]
    while len(first_five) < 5:
        first_five = first_five + '0'
    last_five = str(longitude)[2] + str(longitude)[4:-1]
    while len(last_five) < 5:
        last_five = last_five + '0'
    return first_five + last_five


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
        hdlr = logging.FileHandler(log_file_path)
        hdlr.setLevel(logging.INFO)
        hdlrFormatter = logging.Formatter('%(name)s (%(levelname)s)-%(asctime)s: %(message)s', '%m/%d/%Y  %I:%M:%S %p')
        hdlr.setFormatter(hdlrFormatter)
        log.addHandler(hdlr)
        log.info('Script started: {0}'.format(datetime.datetime.now().strftime('%c %Z')))
    except KeyError:
        errorhandler('Make sure config file exists and contains a log_file variable under "street-segment-identifier".')

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
        print('Processing {0}'.format(vin_number))
        arcpy.MakeFeatureLayer_management(in_features='in_memory/mem_points_sorted', out_layer='mem_vin_points',
                                          where_clause=""" "vin" LIKE '{0}' """.format(vin_number), workspace='in_memory')

        # TODO Test if this works faster by querying insdie of cursor in index_track_points instead of creating feature
        # layer for each vin
        street_seg_identifier(gps_points='mem_vin_points', grid=grid_index, net_decay_constant=30,
                              euclidean_decay_constant=10, max_distance_constant=50)  # TODO place constants in config



