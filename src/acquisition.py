
import stackstac
import geopandas as gpd
import numpy as np
import xarray as xr
import netCDF4
import rioxarray
from rasterio.enums import Resampling

# import earthaccess
# earthaccess.login(strategy="netrc")

from pystac_client import Client
catalog = Client.open("https://cmr.earthdata.nasa.gov/stac/LPCLOUD")


def read_vector(boundary_file,root,cheasapeake=False):
    # city boundary

    boundary = gpd.read_file(root / 'data' / boundary_file)
    boundary_4326 = boundary.to_crs(epsg=4326)
    # bbox
    bbox_utm = tuple(boundary.total_bounds)
    bbox_4326 = tuple(boundary_4326.total_bounds)

    if cheasapeake == True:
        shoreline_boundary = gpd.read_file(root / 'data' / 'bay_shoreline' / 'Chesapeake_Bay_Shoreline_Medium_Resolution.shp')
        shoreline_boundary = shoreline_boundary.to_crs(epsg=26918)
    else:
        shoreline_boundary = None

    return boundary, bbox_utm, bbox_4326, shoreline_boundary

def get_stac_items(catalog_name, from_date,to_date,bbox_utm,bbox_4326,boundary,bands,shoreline=None):

    items = catalog.search(
        bbox=bbox_4326,
        collections=[catalog_name],
        datetime=f"{from_date}/{to_date}"
    ).item_collection()
    len(items)

    # create xarray
    stack = stackstac.stack(
        items,
        epsg=26918,
        resolution=30,
        bounds=bbox_utm,
        assets=bands).where(lambda x: x > 0, other=np.nan)

    # filter for images cloud cover
    lowcloud = stack[stack["eo:cloud_cover"] < 20]
  
    # crop to city boundaries and shoreline
    crop = lowcloud.rio.clip(geometries=boundary.geometry)
    if shoreline is not None:
        crop = crop.rio.clip(geometries=shoreline.geometry, invert=True)

    return crop

def check_data(data,keep):
    drop = data.drop_duplicates(dim='time',keep=keep)
    
    drop.isel(band=3).plot(col='time',col_wrap=5)
    return drop

def make_cloud_mask(a):

        fmask_values= np.unique(a.sel(band='Fmask'))

        all_bits = list()
        for v in fmask_values[:-1]:
                all_bits.append(format(int(v), 'b').zfill(8))

        # cloud = bit index 1
        # cloud shadow = bit index 3
        # snow/ice = bit index 4
        # 0 1 2 3 4 5 6 7 
        # 7 6 5 4 3 2 1 0  bit index

        qa = list()
        for b in all_bits:
                if int(b[3]) + int(b[4]) + int(b[6]) == 0:    # if bits in these positions are all zero then quality is good
                        qa.append('good')
                else:
                        qa.append('poor')

        keep_values = list()
        for (quality, value) in (zip(qa,fmask_values)):
                if quality == 'good':
                        keep_values.append(value)

        return keep_values

def filter_missing_data(data,time_with_missing_data):
    # get number of nas in time with missing data
    threshold = data.isel(band=3,time=time_with_missing_data).isnull().values.sum()
    # get number of nas in each time step
    nan_count = data.isnull().sum(dim=["x", "y"])

    # make boolean arrays showing if timesteps have na count less than threshold
    valid_slices = nan_count.values < threshold

    # get indices of valid timesteps
    keep_index = []
    for i,a in enumerate(valid_slices):
        if a[0] == True:
            keep_index.append(i)

            # filter the DataArray
    filtered = data.isel(time=keep_index)

    return filtered

def mask_and_scale(crop,scale_factor1, sensor, scale_factor2=None):

        mask = make_cloud_mask(crop)

        fmask = crop.sel(band='Fmask')

        masked = crop.where((fmask == mask[0]) | (fmask == mask[1]) | (fmask == mask[2]) | (fmask == mask[3]) |
                        (fmask == mask[4]) | (fmask == mask[5]) | (fmask == mask[6]) | (fmask == mask[7]) |
                        (fmask == mask[8]) | (fmask == mask[9]) | (fmask == mask[10]) | (fmask == mask[11]))

        masked = masked.drop_sel(band='Fmask')  # drop fmask band

        # apply scaling factor
        if sensor == 'landsat':
                scaled1 = masked.isel(band=slice(0,6)) * scale_factor1    
                scaled2 = masked.isel(band=slice(6,None)) * scale_factor2           

                scaled = xr.concat([scaled1,scaled2], dim='band')
        else:
                scaled = masked * scale_factor1

        return scaled

### new version where length of mask values is not hard coded
def mask_and_scale2(crop,scale_factor1, sensor,scale_factor2=None):
        # get values to mask
        mask_values = make_cloud_mask(crop)
        # get fmask band
        fmask = crop.sel(band='Fmask')
        # number of bands
        dim_size = len(crop.band)
        # reshape fmask band to match dimensions of satelite data
        fmask = np.repeat(fmask.values[:, np.newaxis, :,:], dim_size, axis=1)
        # apply fmask to satellite data
        masked = crop.where(np.isin(fmask,mask_values))
        masked = masked.drop_sel(band='Fmask')  # drop fmask band

        # apply scaling factor
        if sensor == 'landsat':
                scaled1 = masked.isel(band=slice(0,6)) * scale_factor1    
                scaled2 = masked.isel(band=slice(6,None)) * scale_factor2           

                scaled = xr.concat([scaled1,scaled2], dim='band')
        else:
                scaled = masked * scale_factor1

        return scaled

# functions to get monthly medians and annual medians of just spectral bands and indices 

def get_shared_bands(landsat_data, sentinel_data):
    
    # combine shared spectral bands
    l_renamed = landsat_data.assign_coords({'band':['blue','green', 'red','nir','sw1','sw2','tir1','tir2']})
    s_renamed = sentinel_data.assign_coords({'band': ['blue','green','red','rededge1','rededge2','rededge3','nir','sw1','sw2']})
    band_list = ['blue','green','red','nir','sw1','sw2']
    combined_bands = l_renamed.sel(band=band_list).combine_first(s_renamed.sel(band=band_list))

    annual_bands = combined_bands.resample(time='YE').median().assign_coords({'time':['annual']})

    combined_bands = combined_bands.resample(time='1MS').median().assign_coords({'time':['april','may','june','july','august','september','october','november']})


    return  combined_bands, annual_bands

def get_unique_bands(l_data,s_data):

    ##### sentinel-specific bands####
    s_unique = s_data.isel(band=slice(3,6)).assign_coords({'band': ['red edge 1', 'red edge 2', 'red edge 3']}).resample(time='1MS').median()
    s_annual = s_unique.resample(time='YE').median().assign_coords({'time':['annual']})

    ##### landsat- specific bands ####
    l_unique = l_data.isel(band=slice(6,8)).assign_coords({'band':['tir1','tir2']}).resample(time='1MS').median()
    l_annual = l_unique.resample(time='YE').median().assign_coords({'time':['annual']})
    all_unique = xr.concat([s_unique, l_unique],dim='band').assign_coords({'time':['april','may','june','july','august','september','october','november']})

    all_annual = xr.concat([s_annual,l_annual],dim='band')

    return all_unique, all_annual

def add_tc_to_annual(filename,city_boundary,root,shoreline,annual1,all_annual):
    tc = rioxarray.open_rasterio(root / 'data' / filename)
    # clip city boundaries
    tc = tc.rio.clip(geometries=city_boundary.geometry)
    if shoreline is not None:
        tc = tc.rio.clip(geometries=shoreline.geometry, invert=True)
    # align to spectral band rasters
    tc_align = tc.rio.reproject_match(annual1, resampling=Resampling.bilinear)  # add resampling = 'bilinear' to fix alignment issue
    # combine rasters
    combined = xr.concat([tc_align,all_annual],dim='band')
    combined = combined.transpose('time','band','y','x')

    return combined

def get_unique_indices(l_data,s_data):

    ##### sentinel-specific variables ####

    combine = list()
    # chlorophyll index red edge (N/RE1)-1
    combine.append((s_data.sel(band='B8A')/s_data.sel(band='B05'))-1)
    # normalized difference red edge index (NDREI) (N - RE1) / (N + RE1)
    combine.append((s_data.sel(band='B8A')-s_data.sel(band='B05'))/(s_data.sel(band='B8A')+s_data.sel(band='B05')))
    # inverted red edge chlorophyll index (IRECI) (RE3 - R) / (RE1 / RE2)
    combine.append((s_data.sel(band='B07')-s_data.sel(band='B04'))/(s_data.sel(band='B05')/s_data.sel(band='B06')))
    # carotenoid index 2 (1.0 / B) - (1.0 / RE1)
    combine.append((1/s_data.sel(band='B02'))-(1/s_data.sel(band='B05')))
    #  Anthocyanin Reflectance Index 2   N * ((1 / G) - (1 / RE1))
    combine.append(s_data.sel(band='B8A')*((1/s_data.sel(band='B03'))-(1/s_data.sel(band='B05'))))
    # red edge ndvi 	(RE2 - RE1)/(RE2 + RE1)
    combine.append((s_data.sel(band='B06')-s_data.sel(band='B05'))/(s_data.sel(band='B06')+s_data.sel(band='B05')))
    # s2 water index (RE1 - S2)/(RE1 + S2)
    combine.append((s_data.sel(band='B05')-s_data.sel(band='B12'))/((s_data.sel(band='B05')+s_data.sel(band='B12'))))

    
    band_list = ['chlorophyll_index_red_edge','ndrei','ireci','carotenoid_index_2','anthocyanin_index_2','red_edge_ndvi','s2_water_index']
    combine = [x.expand_dims(dim={'band':[band_list[i]]}) for i, x in enumerate(combine)]

    annual_sentinel = xr.concat(combine,dim='band').resample(time='YE').median().assign_coords({'time':['annual']})
    
    sentinel_medians = xr.concat(combine,dim='band').resample(time='1MS').median().assign_coords({'time':['april','may','june','july','august','september','october','november']})


    ##### landsat- specific variable ####

    # enhanced built up and bareness index (EBBI) (S1 - N) / (10.0 * ((S1 + T) ** 0.5))
    ebbi = (l_data.sel(band='B06')-l_data.sel(band='B05'))/(10.0*((l_data.sel(band='B06') + l_data.sel(band='B10'))**0.5))
    ebbi = ebbi.expand_dims(dim={'band':['ebbi']})
    landsat_medians = ebbi.resample(time='1MS').median().assign_coords({'time':['april','may','june','july','august','september','october','november']})
    annual_landsat = ebbi.resample(time='YE').median().assign_coords({'time':['annual']})

    month_indices = xr.concat([sentinel_medians, landsat_medians],dim='band')
    annual_indices = xr.concat([annual_sentinel,annual_landsat],dim='band')

    return month_indices, annual_indices

def get_shared_indices(l_data,s_data):
    # calculate and combine shared vegetation indices
    blue = 'B02'
    green = 'B03'
    red = 'B04'

    all_common_bands = list()
    for i in ['landsat','sentinel']:
        combine = list()
        if i == 'landsat':
            nir = 'B05'
            sw1 = 'B06'
            sw2 = 'B07'
            a = l_data
        else:
            nir = 'B8A'
            sw1 = 'B11'
            sw2 = 'B12'
            a = s_data
            
        # tdvi
        combine.append(1.5*((a.sel(band=nir)-a.sel(band=red))/(a.sel(band=nir)**2+a.sel(band=red)+0.5)**0.5))
        # ndwi
        combine.append((a.sel(band=green)-a.sel(band=nir))/(a.sel(band=green)+a.sel(band=nir)))
        # msavi
        combine.append(((2*a.sel(band=nir)+1)-((2*a.sel(band=nir)+1)**2-8*(a.sel(band=nir)-a.sel(band=red)))**0.5)/2)
        #brightness
        combine.append((a.sel(band=blue)*0.3029)+(a.sel(band=green)*0.2786)+(a.sel(band=red)*0.4733)+(a.sel(band=nir)*0.5599)+(a.sel(band=sw1)*0.508)+(a.sel(band=sw2)*0.1872))
        # greenness
        combine.append((a.sel(band=blue)*(-0.2941))+(a.sel(band=green)*(-0.243))+(a.sel(band=red)*(-0.5424))+(a.sel(band=nir)*0.7276)+(a.sel(band=sw1)*0.0713)+(a.sel(band=sw2)*(-0.1608)))
        # wetness
        combine.append((a.sel(band=blue)*0.1511)+(a.sel(band=green)*0.1973)+(a.sel(band=red)*0.3283)+(a.sel(band=nir)*0.3407)+(a.sel(band=sw1)*(-0.7117))+(a.sel(band=sw2)*(-0.4559)))
        # chlorophyll index green
        combine.append((a.sel(band=nir)/a.sel(band=green))-1)

        #perpendicular impervious surface index  (PISI- isa_index)  	0.8192 * B - 0.5735 * N + 0.0750
        combine.append(0.8192*a.sel(band=blue)-0.5735*a.sel(band=nir)+0.0750)
        #carotenoid reflectance index 1
        combine.append((1/a.sel(band=blue))-(1/a.sel(band=green)))
        # chlorophyll vegetation index (N * R) / (G ** 2.0)
        combine.append((a.sel(band=nir)*a.sel(band=red))/(a.sel(band=green)**2))
        # land surface water index (N - S1)/(N + S1)
        combine.append((a.sel(band=nir)-a.sel(band=sw1))/(a.sel(band=nir)+a.sel(band=sw1)))

        # add named band dimension to each array in list
        band_list = ['tdvi','ndwi','msavi','brightness','greenness','wetness','chlorophyll_index_green','isa_index','carotenoid_index_1','chlorophyll_veg_index','lswi']
        combine = [x.expand_dims(dim={'band':[band_list[i]]}) for i, x in enumerate(combine)]

        common_bands = xr.concat(combine,dim='band')

        all_common_bands.append(common_bands) 

    combined_indices = all_common_bands[0].combine_first(all_common_bands[1])  # merge landsat and sentinel time series
    annual_indices = combined_indices.resample(time='YE').median().assign_coords({'time':['annual']})

    month_indices = combined_indices.resample(time='1MS').median().assign_coords({'time':['april','may','june','july','august','september','october','november']})

    return  month_indices, annual_indices

def concatenate_and_save_data(annual1,annual2,monthly1,monthly2,filename,root,tc_filename=None,city_boundary=None,add_canopy=False,shoreline=None):
    all_annual = xr.concat([annual1,annual2],dim='band').assign_coords({'time':['annual']})

    # add tc to annual bands raster
    if add_canopy == True:
        all_annual = add_tc_to_annual(tc_filename,city_boundary,root,shoreline,annual1,all_annual=all_annual)
    # combine monthly bands
    concat_bands1 = xr.concat([monthly1,monthly2],dim='band')
    # combine monthly and annual
    concat_bands2 = xr.concat([concat_bands1,all_annual],dim='time')

    # save to disk
    concat_bands2.to_netcdf(root / 'data' / f'{filename}.nc')

    return concat_bands2 


##### archive #####


def get_shared_variables(landsat_data, sentinel_data):
    
    # calculate and combine shared vegetation indices
    blue = 'B02'
    green = 'B03'
    red = 'B04'

    all_common_bands = list()
    for i in ['landsat','sentinel']:
        combine = list()
        if i == 'landsat':
            nir = 'B05'
            sw1 = 'B06'
            sw2 = 'B07'
            a = landsat_data
        else:
            nir = 'B8A'
            sw1 = 'B11'
            sw2 = 'B12'
            a = sentinel_data
            
        # tdvi
        combine.append(1.5*((a.sel(band=nir)-a.sel(band=red))/(a.sel(band=nir)**2+a.sel(band=red)+0.5)**0.5))
        # ndwi
        combine.append((a.sel(band=green)-a.sel(band=nir))/(a.sel(band=green)+a.sel(band=nir)))
        # msavi
        combine.append(((2*a.sel(band=nir)+1)-((2*a.sel(band=nir)+1)**2-8*(a.sel(band=nir)-a.sel(band=red)))**0.5)/2)
        # perpendicular impervious surface index  (PISI- isa_index)  	0.8192 * B - 0.5735 * N + 0.0750
        combine.append(0.8192*a.sel(band=blue)-0.5735*a.sel(band=nir)+0.0750)
        #brightness
        combine.append((a.sel(band=blue)*0.3029)+(a.sel(band=green)*0.2786)+(a.sel(band=red)*0.4733)+(a.sel(band=nir)*0.5599)+(a.sel(band=sw1)*0.508)+(a.sel(band=sw2)*0.1872))
        # greenness
        combine.append((a.sel(band=blue)*(-0.2941))+(a.sel(band=green)*(-0.243))+(a.sel(band=red)*(-0.5424))+(a.sel(band=nir)*0.7276)+(a.sel(band=sw1)*0.0713)+(a.sel(band=sw2)*(-0.1608)))
        # wetness
        combine.append((a.sel(band=blue)*0.1511)+(a.sel(band=green)*0.1973)+(a.sel(band=red)*0.3283)+(a.sel(band=nir)*0.3407)+(a.sel(band=sw1)*(-0.7117))+(a.sel(band=sw2)*(-0.4559)))
        # chlorophyll index green
        combine.append((a.sel(band=nir)/a.sel(band=green))-1)
        # carotenoid reflectance index 1
        combine.append((1/a.sel(band=blue))-(1/a.sel(band=green)))
        # chlorophyll vegetation index (N * R) / (G ** 2.0)
        combine.append((a.sel(band=nir)*a.sel(band=red))/(a.sel(band=green)**2))
        # land surface water index (N - S1)/(N + S1)
        combine.append((a.sel(band=nir)-a.sel(band=sw1))/(a.sel(band=nir)+a.sel(band=sw1)))


        band_list = ['tdvi','ndwi','msavi','isa_index','brightness','greenness','wetness','chlorophyll_index_green','carotenoid_index_1','chlorophyll_veg_index','lswi']
        combine = [x.expand_dims(dim={'band':[band_list[i]]}) for i, x in enumerate(combine)]

        common_bands = xr.concat(combine,dim='band')

        all_common_bands.append(common_bands) 

    combined_indices = all_common_bands[0].combine_first(all_common_bands[1])  # merge landsat and sentinel time series
    combined_indices = combined_indices.resample(time='3MS').median()

    # combine shared spectral bands
    l_renamed = landsat_data.assign_coords({'band':['blue','green', 'red','nir','sw1','sw2','tir1','tir2']})
    s_renamed = sentinel_data.assign_coords({'band': ['blue','green','red','rededge1','rededge2','rededge3','nir','sw1','sw2']})
    band_list = ['blue','green','red','nir','sw1','sw2']
    combined_bands = l_renamed.sel(band=band_list).combine_first(s_renamed.sel(band=band_list))

    combined_bands = combined_bands.resample(time='3MS').median()

    all_shared_variables = xr.concat([combined_bands,combined_indices],dim='band').assign_coords({'time':['spring','summer','fall','winter']})

    return  all_shared_variables

def get_unique_variables(s_data, l_data):

    ##### sentinel-specific variables ####

    combine = list()
    # chlorophyll index red edge (N/RE1)-1
    combine.append((s_data.sel(band='B8A')/s_data.sel(band='B05'))-1)
    # normalized difference red edge index (NDREI) (N - RE1) / (N + RE1)
    combine.append((s_data.sel(band='B8A')-s_data.sel(band='B05'))/(s_data.sel(band='B8A')+s_data.sel(band='B05')))
    # inverted red edge chlorophyll index (IRECI) (RE3 - R) / (RE1 / RE2)
    combine.append((s_data.sel(band='B07')-s_data.sel(band='B04'))/(s_data.sel(band='B05')/s_data.sel(band='B06')))
    # carotenoid index 2 (1.0 / B) - (1.0 / RE1)
    combine.append((1/s_data.sel(band='B02'))-(1/s_data.sel(band='B05')))
    #  Anthocyanin Reflectance Index 2   N * ((1 / G) - (1 / RE1))
    combine.append(s_data.sel(band='B8A')*((1/s_data.sel(band='B03'))-(1/s_data.sel(band='B05'))))
    # red edge ndvi 	(RE2 - RE1)/(RE2 + RE1)
    combine.append((s_data.sel(band='B06')-s_data.sel(band='B05'))/(s_data.sel(band='B06')+s_data.sel(band='B05')))
    # s2 water index (RE1 - S2)/(RE1 + S2)
    combine.append((s_data.sel(band='B05')-s_data.sel(band='B12'))/((s_data.sel(band='B05')+s_data.sel(band='B12'))))

    
    band_list = ['chlorophyll_index_red_edge','ndrei','ireci','carotenoid_index_2','anthocyanin_index_2','red_edge_ndvi','s2_water_index']
    combine = [x.expand_dims(dim={'band':[band_list[i]]}) for i, x in enumerate(combine)]
    
    all_indices_medians = xr.concat(combine,dim='band').resample(time='3MS').median()

    spectral_bands_medians = s_data.isel(band=slice(3,6)).assign_coords({'band': ['red edge 1', 'red edge 2', 'red edge 3']}).resample(time='3MS').median()
    
    s_unique = xr.concat([all_indices_medians,spectral_bands_medians],dim='band' ).assign_coords({'time':['spring','summer','fall','winter']})

    ##### landsat- specific variable ####

    # enhanced built up and bareness index (EBBI) (S1 - N) / (10.0 * ((S1 + T) ** 0.5))
    ebbi = (l_data.sel(band='B06')-l_data.sel(band='B05'))/(10.0*((l_data.sel(band='B06') + l_data.sel(band='B10'))**0.5))
    ebbi_medians = ebbi.expand_dims(dim={'band':['ebbi']}).resample(time='3MS').median()
    spectral_bands_medians = l_data.isel(band=slice(6,8)).assign_coords({'band':['tir1','tir2']}).resample(time='3MS').median()
    l_unique = xr.concat([ebbi_medians,spectral_bands_medians],dim='band').assign_coords({'time':['spring','summer','fall','winter']})

    

    all_unique = xr.concat([s_unique, l_unique],dim='band')


    return all_unique