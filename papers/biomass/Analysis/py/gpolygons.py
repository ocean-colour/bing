import earthaccess
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.ops import orient
import geopandas as gpd
from matplotlib.patches import Polygon as MPLPolygon
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from remote_sensing.download import earthaccess as rs_ea


def plot_spatial_extent(granule, granule_title="Data Granule"):
    """
    Plot the spatial extent using matplotlib and cartopy.
    
    Parameters:
    -----------
    granule : dict
        Raw spatial information
    granule_title : str
        Title for the plot
    """
    spatial_info = rs_ea.extract_spatial_extent(granule)
    geometries = rs_ea.create_shapely_geometries(spatial_info, 
                                                 fix_antimeridian=True)

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