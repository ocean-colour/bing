import earthaccess
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.ops import orient
import geopandas as gpd
from matplotlib.patches import Polygon as MPLPolygon
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def extract_spatial_extent(granule):
    """
    Extract spatial extent information from an earthaccess DataGranule object.
    
    Parameters:
    -----------
    granule : earthaccess.results.DataGranule
        The granule object to extract spatial extent from
        
    Returns:
    --------
    dict : Dictionary containing extracted spatial information
    """
    spatial_info = {
        'bounding_rectangles': [],
        'gpolygons': [],
        'points': [],
        'lines': []
    }
    
    # Check if spatial extent exists
    if 'SpatialExtent' not in granule.get('umm', {}):
        print("No spatial extent found in granule metadata")
        return spatial_info
    
    spatial_extent = granule['umm']['SpatialExtent']
    
    # Extract horizontal spatial domain
    if 'HorizontalSpatialDomain' in spatial_extent:
        horizontal_domain = spatial_extent['HorizontalSpatialDomain']
        
        if 'Geometry' in horizontal_domain:
            geometry = horizontal_domain['Geometry']
            
            # Extract GPolygons
            if 'GPolygons' in geometry:
                for gpolygon in geometry['GPolygons']:
                    polygon_info = {
                        'boundary_points': [],
                        'exclusive_zones': []
                    }
                    
                    # Extract boundary points
                    if 'Boundary' in gpolygon:
                        boundary = gpolygon['Boundary']
                        if 'Points' in boundary:
                            points = [(point['Longitude'], point['Latitude']) 
                                    for point in boundary['Points']]
                            polygon_info['boundary_points'] = points
                    
                    # Extract exclusive zones (holes)
                    if 'ExclusiveZone' in gpolygon:
                        if 'Boundaries' in gpolygon['ExclusiveZone']:
                            for boundary in gpolygon['ExclusiveZone']['Boundaries']:
                                if 'Points' in boundary:
                                    hole_points = [(point['Longitude'], point['Latitude']) 
                                                 for point in boundary['Points']]
                                    polygon_info['exclusive_zones'].append(hole_points)
                    
                    spatial_info['gpolygons'].append(polygon_info)
            
            # Extract BoundingRectangles
            if 'BoundingRectangles' in geometry:
                for rect in geometry['BoundingRectangles']:
                    spatial_info['bounding_rectangles'].append({
                        'west': rect.get('WestBoundingCoordinate'),
                        'east': rect.get('EastBoundingCoordinate'), 
                        'north': rect.get('NorthBoundingCoordinate'),
                        'south': rect.get('SouthBoundingCoordinate')
                    })
            
            # Extract Points
            if 'Points' in geometry:
                for point in geometry['Points']:
                    spatial_info['points'].append((
                        point['Longitude'], 
                        point['Latitude']
                    ))
            
            # Extract Lines
            if 'Lines' in geometry:
                for line in geometry['Lines']:
                    if 'Points' in line:
                        line_points = [(point['Longitude'], point['Latitude']) 
                                     for point in line['Points']]
                        spatial_info['lines'].append(line_points)
    
    return spatial_info

def create_shapely_geometries(spatial_info):
    """
    Convert spatial extent information to Shapely geometry objects.
    
    Parameters:
    -----------
    spatial_info : dict
        Spatial information dictionary from extract_spatial_extent
        
    Returns:
    --------
    dict : Dictionary containing Shapely geometry objects
    """
    geometries = {
        'polygons': [],
        'rectangles': [],
        'points': [],
        'lines': []
    }
    
    # Convert GPolygons to Shapely Polygons
    for gpolygon in spatial_info['gpolygons']:
        if gpolygon['boundary_points']:
            # Create exterior ring
            exterior = gpolygon['boundary_points']
            
            # Create holes (exclusive zones)
            holes = gpolygon['exclusive_zones'] if gpolygon['exclusive_zones'] else None
            
            try:
                # Ensure proper orientation (counter-clockwise for exterior)
                polygon = Polygon(exterior, holes=holes)
                # Orient exterior counter-clockwise, holes clockwise
                oriented_polygon = orient(polygon)
                geometries['polygons'].append(oriented_polygon)
            except Exception as e:
                print(f"Error creating polygon: {e}")
    
    # Convert BoundingRectangles to Shapely Polygons
    for rect in spatial_info['bounding_rectangles']:
        if all(coord is not None for coord in [rect['west'], rect['east'], 
                                              rect['north'], rect['south']]):
            # Create rectangle coordinates (counter-clockwise)
            rect_coords = [
                (rect['west'], rect['south']),   # SW
                (rect['east'], rect['south']),   # SE  
                (rect['east'], rect['north']),   # NE
                (rect['west'], rect['north']),   # NW
                (rect['west'], rect['south'])    # Close
            ]
            geometries['rectangles'].append(Polygon(rect_coords))
    
    # Convert Points to Shapely Points
    for point in spatial_info['points']:
        geometries['points'].append(Point(point))
    
    # Convert Lines to Shapely LineStrings
    from shapely.geometry import LineString
    for line in spatial_info['lines']:
        if len(line) >= 2:
            geometries['lines'].append(LineString(line))
    
    return geometries

def plot_spatial_extent(spatial_info, geometries, granule_title="Data Granule"):
    """
    Plot the spatial extent using matplotlib and cartopy.
    
    Parameters:
    -----------
    spatial_info : dict
        Raw spatial information
    geometries : dict  
        Shapely geometry objects
    granule_title : str
        Title for the plot
    """
    fig = plt.figure(figsize=(12, 8))
    
    # Determine bounds for the plot
    all_coords = []
    for gpolygon in spatial_info['gpolygons']:
        all_coords.extend(gpolygon['boundary_points'])
    for rect in spatial_info['bounding_rectangles']:
        if all(coord is not None for coord in [rect['west'], rect['east'], 
                                              rect['north'], rect['south']]):
            all_coords.extend([
                (rect['west'], rect['south']),
                (rect['east'], rect['north'])
            ])
    all_coords.extend(spatial_info['points'])
    
    if not all_coords:
        print("No coordinates found to plot")
        return fig
    
    # Calculate bounds with buffer
    lons = [coord[0] for coord in all_coords]
    lats = [coord[1] for coord in all_coords]
    lon_buffer = (max(lons) - min(lons)) * 0.1 if max(lons) != min(lons) else 1
    lat_buffer = (max(lats) - min(lats)) * 0.1 if max(lats) != min(lats) else 1
    
    # Create map with cartopy
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([
        min(lons) - lon_buffer, max(lons) + lon_buffer,
        min(lats) - lat_buffer, max(lats) + lat_buffer
    ], ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, alpha=0.8)
    ax.add_feature(cfeature.BORDERS, alpha=0.8)
    ax.add_feature(cfeature.OCEAN, alpha=0.3)
    ax.add_feature(cfeature.LAND, alpha=0.3)
    ax.gridlines(draw_labels=True, alpha=0.5)
    
    # Plot polygons
    for i, polygon in enumerate(geometries['polygons']):
        x, y = polygon.exterior.xy
        ax.plot(x, y, 'r-', linewidth=2, label=f'GPolygon {i+1}' if i == 0 else "")
        ax.fill(x, y, 'red', alpha=0.2)
        
        # Plot holes
        for hole in polygon.interiors:
            x, y = hole.xy
            ax.plot(x, y, 'b-', linewidth=1)
            ax.fill(x, y, 'white', alpha=0.8)
    
    # Plot bounding rectangles
    for i, rect_poly in enumerate(geometries['rectangles']):
        x, y = rect_poly.exterior.xy
        ax.plot(x, y, 'g--', linewidth=2, label=f'Bounding Box {i+1}' if i == 0 else "")
    
    # Plot points
    for i, point in enumerate(geometries['points']):
        ax.plot(point.x, point.y, 'ko', markersize=8, 
               label='Points' if i == 0 else "")
    
    # Plot lines
    for i, line in enumerate(geometries['lines']):
        x, y = line.xy
        ax.plot(x, y, 'm-', linewidth=2, label=f'Line {i+1}' if i == 0 else "")
    
    plt.title(f'{granule_title}\nSpatial Extent Visualization')
    plt.legend(loc='upper right')
    
    return fig

def demonstrate_spatial_extent_extraction():
    """
    Demonstrate the spatial extent extraction process.
    """
    print("Demonstrating spatial extent extraction from earthaccess DataGranule...")
    
    # Authenticate with earthaccess
    try:
        auth = earthaccess.login()
        print("✓ Successfully authenticated with earthaccess")
    except Exception as e:
        print(f"Authentication failed: {e}")
        return
    
    # Search for data with spatial extent
    try:
        results = earthaccess.search_data(
            short_name="ATL06",  # ICESat-2 data with good spatial extent
            temporal=("2020-03-01", "2020-03-02"),
            bounding_box=(-134.7, 58.9, -133.9, 59.2),
            count=1
        )
        
        if not results:
            print("No granules found")
            return
            
        granule = results[0]
        print(f"✓ Found granule: {granule.get('meta', {}).get('concept-id', 'Unknown')}")
        
    except Exception as e:
        print(f"Search failed: {e}")
        return
    
    # Extract spatial extent
    spatial_info = extract_spatial_extent(granule)
    
    print("\n=== Spatial Extent Summary ===")
    print(f"GPolygons found: {len(spatial_info['gpolygons'])}")
    print(f"Bounding rectangles found: {len(spatial_info['bounding_rectangles'])}")
    print(f"Points found: {len(spatial_info['points'])}")
    print(f"Lines found: {len(spatial_info['lines'])}")
    
    # Show detailed information
    if spatial_info['gpolygons']:
        print("\n=== GPolygon Details ===")
        for i, gpolygon in enumerate(spatial_info['gpolygons']):
            print(f"GPolygon {i+1}:")
            print(f"  Boundary points: {len(gpolygon['boundary_points'])}")
            if gpolygon['boundary_points']:
                print(f"  First few points: {gpolygon['boundary_points'][:3]}")
            print(f"  Exclusive zones (holes): {len(gpolygon['exclusive_zones'])}")
    
    if spatial_info['bounding_rectangles']:
        print("\n=== Bounding Rectangle Details ===")
        for i, rect in enumerate(spatial_info['bounding_rectangles']):
            print(f"Rectangle {i+1}: West={rect['west']}, East={rect['east']}, "
                  f"North={rect['north']}, South={rect['south']}")
    
    # Create Shapely geometries
    geometries = create_shapely_geometries(spatial_info)
    
    # Plot the spatial extent
    fig = plot_spatial_extent(spatial_info, geometries, 
                             f"Granule {granule.get('meta', {}).get('concept-id', 'Unknown')}")
    
    plt.tight_layout()
    plt.show()
    
    return spatial_info, geometries

# Example usage
if __name__ == "__main__":
    # Run the demonstration
    demonstrate_spatial_extent_extraction()