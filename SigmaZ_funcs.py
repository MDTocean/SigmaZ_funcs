import matplotlib.pyplot as plt
from matplotlib import gridspec
import xarray as xr
import numpy as np
import sectionate
from glob import glob
import momlevel
from numba import njit
from xhistogram.xarray import histogram
import os

def SigmaZ_diag_ReadData(
    dir_vars,
    dir_grid,
    section_node_lons,
    section_node_lats,
    file_str_identifier,
    theta_var="thetao",
    salt_var="so",
    rho_var="",
    z_layer_var="z_l",
    z_inter_var="z_i",
    u_transport_var="umo",
    v_transport_var="vmo",
    ref_pres=0,
    time_limits=[],
    x_hpoint_1Dvar="xh",
    x_qpoint_1Dvar="xq",
    y_hpoint_1Dvar="yh",
    y_qpoint_1Dvar="yq",
    time_var="time",
    lons_tpoint="geolon",
    lats_tpoint="geolat",
    lons_cpoint="geolon_c",
    lats_cpoint="geolat_c",
    dir_deptho="",
    decode_times_flag=True,
    mom_gridwidth_algo=False,
    supergrid_flag=False,
    calc_transp=False,
    dmgetout=False,
    zarr_dir='',
):
    """Read model data to be used for calculating a SigmaZ diagram (See Zhang and Thomas, 2021, Nat Comm Earth Env.) along a specified cross section. See description of the inputs below -- there is also a readme with more meandering thoughts on some of the choices of these input types. 
    
    Args:
        dir_vars (string): directory containing the output variable files. 
        dir_grid (string): directory cotaining the grid info file
        section_node_lons (vector): 1D vector of longitude nodes (must be same length as section_node_lats) that define a cross section. For use in Raphael Dussin's sectionate tool, which finds all grid cell locations along lines connecting each lon/lat node E.g.: [60.3000, 58.8600, 58.0500, 58.0000, 56.5000]
        section_node_lats (vector): 1D vector of latitude nodes (must be same length as section_node_lons) that define the cross section. E.g [-44.9000, -30.5400, -28.0000, -14.7000, -5.9300]
        file_str_identifier (string): string that identifies a particular output file or set of output files. E.g. ".0196*", for all strings containing ".0196", or "*.0[1-2]*" for years 100-200. The script assumes that the variables are kept in separate files; if all variables are in a single netcdf file then specify .nc in the file_str_identifier e.g. "*/*.ocean_month.nc" will find all ocean_month.nc files within all folders. 
        theta_var (string): variable name for reading potential temperature from file. If an empty string is given, it will not read temperature
        salt_var (string): variable name for reading salt from file. If an empty string is given, it will not read salt
        rho_var (string): variable name for reading rho from file -- if an empty string is given, but strings are given for temp and salt, then the script will calculate rho from T and S using a reference of "ref_pres"
        z_layer_var (string): variable name for the depth coordinate at the grid centre
        z_inter_var (string): variable name for the depth coordinate at the grid interface
        u_transport_var (string): variable name for the zonal transport
        v_transport_var (string): variable name for the meridional transport
        time_limits (vector): a 2-element vector of the type [time_start,time_limit], using the same units as in the data, that allows the data to be limited within these time margins. If using decode_times=True then the vector can contain year strings  (e.g. time_limits=['2014','2020']), otherwise it can also contain index values (e.g. time_limits=[8000,10000])
        ref_pres (scalar): Reference pressure in dB
        x_hpoint_1Dvar (string) etc: coordinate names of the 1D lon and lat q and h points
        lons_tpoint (string) etc: coordinate names for the 2D lon and lat grid at the grid centre
        lons_cpoint (string) etc: coordinate names for the 2D lon and lat grid at the grid corner
        time_var (string): coordinate name of the time dimension
        decode_times_flag (logical): use metadata to convert time to a date
        mom_gridwidth_algo (logical): use mom's gridwidth variables (called dxCv etc). This uses my own branch of Raphael Dussin's sectionate tool (called dev_MDT), which can be found on my github (MDTocean). 
        supergrid_flag (logical): Set to True if dir_grid points towards the supergrid (twice the size of the regular grid). The script will create a regular grid from it. 
        calc_transp (logical): Set to True if u_transport_var and v_transport_var are velocities. The script will then calculate the partial cells and determine cell face area along the specified cross section, and multiply the velocity by the area. NOTE THAT THIS IS WORK IN PROGRESS, AND NOT YET WORKING PROPERLY. FEEL FREE TO FIX IT!!!
        dmgetout(logical): GFDL-specific flag that exits the function and outputs the dmget commands that first need to be run in a terminal to retrieve the data. 
        zarr_dir ('string'): If an empty string then it will do nothing. If given a directory, the code will: 1) save the data as equivalent zarr data to that directory if the zarr data doesn't already exist; 2) read the data from that zarr directory if the zarr data does already exist there. Zarr files will be saved separately for every file read in, and saved at '/zarr_dir/dir_base/dir_vars/' (where dir_base etc are also given above as inputs). 

    output:
        dsT (xarray dataset):  output from Raphael Dussin's sectionate tool, containing transport and grid information along the specified cross-section
        transp (ND xarray): cross section of transports perpendicular to the specified cross section
        rho (ND xarray): cross section of density along the specified cross section
        xsec (1D array): cross-sectional lon coordinates 
        ysec (1D array): cross-sectional lat coordinates 
        cell_area (2D xarray): cross-sectional cell area
        so (ND xarray): cross section of salinity along the specified cross section
        thetao (ND xarray): cross section of temperature along the specified cross section
        section_gridwidth (1D xarray): gridwidth between each section coordinate
        grid_region (xarrat dataset): grid data of the region surrounding the cross-section (useful only really for plotting the section location)

    """
    
    ########################
    ### create a list of files to load:
    
    if ".nc" in file_str_identifier:
        ### If all variables are in a single .nc file, read variables from there:
        vars_str=dir_vars+"*"+file_str_identifier
        files_timestep=glob(f"{vars_str}")
        if dmgetout:
            print(f"{'dmget '+vars_str+' &'}")
            return [],[],[],[],[],[],[],[],[],[]
    else:
        ### If the variables are in separate files, read those instead:
        umo_vars_str=dir_vars+"*"+file_str_identifier+"*."+u_transport_var+".nc"
        vmo_vars_str=dir_vars+"*"+file_str_identifier+"*."+v_transport_var+".nc"
        files_timestep=glob(f"{umo_vars_str}")
        files_timestep+=glob(f"{vmo_vars_str}")
        if dmgetout:
            print(f"{'dmget '+umo_vars_str+' &'}")
            print(f"{'dmget '+vmo_vars_str+' &'}")
        if rho_var:
            rho_vars_str=dir_vars+"*"+file_str_identifier+"*."+rho_var+".nc"
            files_timestep+=glob(f"{rho_vars_str}")
            if dmgetout: print(f"{'dmget '+rho_vars_str+' &'}")
        if theta_var:
            theta_vars_str=dir_vars+"*"+file_str_identifier+"*."+theta_var+".nc"
            files_timestep+=glob(f"{theta_vars_str}")
            if dmgetout: print(f"{'dmget '+theta_vars_str+' &'}")
        if salt_var:
            salt_vars_str=dir_vars+"*"+file_str_identifier+"*."+salt_var+".nc"
            files_timestep+=glob(f"{salt_vars_str}")
            if dmgetout: print(f"{'dmget '+salt_vars_str+' &'}")
        if dmgetout: return [],[],[],[],[],[],[],[],[],[]
        else:
            print(f"{'error: variable-strings must be provided either for rho or for T&S'}")
    
    ########################
    ### If a zarr_dir is given, modify the list of files to read to/write from there instead:

    if zarr_dir:
        for count, filename in enumerate(files_timestep):
            files_timestep[count] = zarr_dir+filename+'.zarr'
            if os.path.isdir(zarr_dir+filename+'.zarr')==False:
                ds_filename=xr.open_dataset(filename,decode_times=decode_times_flag, chunks={x_hpoint_1Dvar : 100, x_qpoint_1Dvar : 100, y_hpoint_1Dvar : 100, y_qpoint_1Dvar : 100})
                if ".nc" in file_str_identifier:
                    variables2keep=[u_transport_var,v_transport_var,theta_var,salt_var,rho_var,z_layer_var,z_inter_var]
                    variables2keep = [i for i in variables2keep if i]
                    ds_filename=ds_filename[variables2keep]
                ds_filename.to_zarr(zarr_dir+filename+'.zarr')
    
    ########################
    ### Read the data from the list of files:
    
    ds = xr.open_mfdataset(files_timestep, decode_times=decode_times_flag)
    grid = xr.open_dataset(dir_grid)
    
    ########################
    ### If supergrid_flag is True, calculate the regular grid from the supergrid
    
    if supergrid_flag:
        grid = use_supergrid(grid,x_hpoint_1Dvar,x_qpoint_1Dvar,y_hpoint_1Dvar,y_qpoint_1Dvar)
        
        
    #######################
    ### If the 1D coordinate names are not xh,xq, etc, then rename them to x_hpoint_1Dvar, x_qpoint_1Dvar etc (The sectionate tool requries it):
    
    if y_hpoint_1Dvar != "yh":
        ds = ds.rename({y_hpoint_1Dvar: 'yh'})
        grid = grid.rename({y_hpoint_1Dvar: 'yh'})
    if y_qpoint_1Dvar != "yq":
        ds = ds.rename({y_qpoint_1Dvar: 'yq'})
        grid = grid.rename({y_qpoint_1Dvar: 'yq'})
    if x_hpoint_1Dvar != "xh":
        ds = ds.rename({x_hpoint_1Dvar: 'xh'})
        grid = grid.rename({x_hpoint_1Dvar: 'xh'})
    if x_qpoint_1Dvar != "xq":
        ds = ds.rename({x_qpoint_1Dvar: 'xq'})
        grid = grid.rename({x_qpoint_1Dvar: 'xq'})   
    if time_var != "time":
        ds = ds.rename({time_var: 'time'})
        grid = grid.rename({time_var: 'time'})
    
    #########################
    ### If there are not the same number of dimensions of yq and yh in the variables dataset, cut off the extra dimension (i.e. it needs to be assymetrical): 
    
    if len(ds['yq'])==len(ds['yh'])+1:
        ds = ds.isel( yq = slice(1,len(ds['yq']))  )
    elif len(ds['yq'])==len(ds['yh']):
        'this is fine'
    else:
        raise Exception("yq must be more positive than yh or be 1 element larger than yh")
    if len(ds['xq'])==len(ds['xh'])+1:
        ds = ds.isel( xq = slice(1,len(ds['xq']))  )
    elif len(ds['xq'])==len(ds['xh']):
        'this is fine'
    else:
        raise Exception("xq must be more positive than xh or be 1 element larger than xh")
    
    ### If there are not the same number of dimensions of yq and yh in the grid dataset, cut off the extra dimension:
    if len(grid['yq'])==len(grid['yh'])+1:
        grid = grid.isel( yq = slice(1,len(ds['yq']))  )
    elif len(grid['yq'])==len(grid['yh']):
        'this is fine'
    else:
        raise Exception("yq must be more positive than yh or be 1 element larger than yh")
    if len(grid['xq'])==len(grid['xh'])+1:
        grid = grid.isel( xq = slice(1,len(ds['xq']))  )
    elif len(grid['xq'])==len(grid['xh']):
        'this is fine'
    else:
        raise Exception("xq must be more positive than xh or be 1 element larger than xh")
    
    
    #########################
    ### The supergrid can be a bit odd, so if using it then this ensures that the variable and grid datasets have exactly the same lon and lat:
    if supergrid_flag:
        ds['xq']=grid['xq']
        ds['xh']=grid['xh']
        ds['yq']=grid['yq']
        ds['yh']=grid['yh']
    
    
    ########################
    ### Reduce the spatial domain to fit around the chosen section coordinates (with a 10 deg buffer around it)
    lat_range_min=np.abs(grid.yh-(min(section_node_lats)-10)).argmin()
    lat_range_max=np.abs(grid.yh-(max(section_node_lats)+10)).argmin()
    lon_range_min=np.abs(grid.xh-(min(section_node_lons)-10)).argmin()
    lon_range_max=np.abs(grid.xh-(max(section_node_lons)+10)).argmin()
    ds_subpolar = ds.sel(yq=slice(ds.yq[lat_range_min],ds.yq[lat_range_max]),
                             xh=slice(ds.xh[lon_range_min],ds.xh[lon_range_max]),
                             yh=slice(ds.yh[lat_range_min],ds.yh[lat_range_max]),
                             xq=slice(ds.xq[lon_range_min],ds.xq[lon_range_max]))
    grid_region = grid.sel(yq=slice(grid.yq[lat_range_min],grid.yq[lat_range_max]),
                             xh=slice(grid.xh[lon_range_min],grid.xh[lon_range_max]),
                             yh=slice(grid.yh[lat_range_min],grid.yh[lat_range_max]),
                             xq=slice(grid.xq[lon_range_min],grid.xq[lon_range_max]))
    if time_limits:
        ds_subpolar = ds_subpolar.sel(time=slice(time_limits[0],time_limits[1]))
    
    ########################
    ### Run Raf's sectionate tool to grid section coordinates
    isec, jsec, xsec, ysec = sectionate.create_section_composite(grid_region['geolon_c'],
                                                                 grid_region['geolat_c'],
                                                                 section_node_lons,
                                                                 section_node_lats)    
    
    #######################
    ### Uncomment this to use the sectionate's built in functionality "find_offset_corner". I found it problematic, so I am just setting everything to zero for now, which works at least for MOM6. 
    
    corner_offset1=0  
    corner_offset2=0 
    #[corner_offset1,corner_offset2]=sectionate.find_offset_center_corner(grid_region[lons_tpoint], grid_region[lats_tpoint],grid_region[lons_cpoint], grid_region[lats_cpoint])

    
    #####################
    ### Use sectionate with the xarray dataset of variables to get the transport along the chosen cross-section:
    
    dsT = sectionate.MOM6_normal_transport(ds_subpolar, isec, jsec,utr=u_transport_var,vtr=v_transport_var,layer=z_layer_var,interface=z_inter_var,
                                           offset_center_x=corner_offset1,offset_center_y=corner_offset2,old_algo=True)
    transp=dsT.uvnormal
    
    #####################
    ### Use sectionate with the xarray dataset of variables to get T,S and rho (if specified in the inputs) along the chosen cross-section:

    thetao=[]
    so=[]
    if theta_var:
        thetao = sectionate.MOM6_extract_hydro(ds_subpolar[theta_var], isec, jsec,offset_center_x=corner_offset1,offset_center_y=corner_offset2)
    if salt_var:
        so = sectionate.MOM6_extract_hydro(ds_subpolar[salt_var], isec, jsec,offset_center_x=corner_offset1,offset_center_y=corner_offset2)
    if rho_var:
        rho = sectionate.MOM6_extract_hydro(ds_subpolar[rho_var], isec, jsec,offset_center_x=corner_offset1,offset_center_y=corner_offset2)-1000
    else:
        if theta_var and salt_var:
            rho = momlevel.derived.calc_rho(thetao,so,ref_pres*1e4)-1000
        else:
            error('Either a density variable must be given, or the salt and temperature variables must be given (or all three)')
    
    rho = rho.rename('rho')
    
    
    ######################
    ### If mom_gridwidth_algo is True, use DxCv etc to extract the along-cell length of all cells along the specified cross-section. Otherwise, approximate it: 
    
    if mom_gridwidth_algo:
        ds_grid_width = sectionate.sectionate_gridwidth(grid_region,ds_subpolar,isec,jsec,interface=z_inter_var,layer=z_layer_var)
        cell_area = ds_grid_width.section_cellarea
        section_gridwidth = ds_grid_width.section_gridwidth
    else:
        earth_radius=6371000  # in km
        section_gridwidth=earth_radius*sectionate.distance_on_unit_sphere(ysec[0:-1],xsec[0:-1],ysec[1:],xsec[1:])
        depth_diff=np.diff(dsT[z_inter_var])
        cell_area=np.repeat(np.expand_dims(section_gridwidth, axis = 0),np.shape(dsT[z_layer_var])[0],axis=0)*np.repeat(np.expand_dims(depth_diff, axis = 1),np.shape(section_gridwidth)[0],axis=1)
        cell_area=xr.DataArray(data=cell_area, coords={z_layer_var : transp[z_layer_var], 'sect' : transp['sect']}, dims=(z_layer_var,'sect')).drop_vars('sect')
    
    ######################
    ### If calc_transp is True, first determine the cell area (including calculating the partial cells), and multiply by the velocities to get transport: 
    
    if calc_transp:
        partial_cell_section = calc_partial_cells(dir_deptho,ds,grid,section_node_lons,section_node_lats,x_hpoint_1Dvar,
                                                  x_qpoint_1Dvar,y_hpoint_1Dvar,y_qpoint_1Dvar,u_transport_var,v_transport_var)
        
        partial_cells = partial_cell_section.section_gridwidth
        transp=transp*cell_area*partial_cells
        
    #####################
    
    return dsT, transp, rho, xsec, ysec, cell_area, so, thetao, section_gridwidth, grid_region


def SigmaZ_diag_ComputeDiag(transp,rho,rholims,depth,rebin_depth=[],nrho=500,z_layer_var='z_l',dist_var='sect',time_var='time',numpy_algo=False):
    """ calculate a SigmaZ diagram from z-space cross-sections of transport and rho (e.g. model output read by the SigmaZ_diag_ReadData function, or other cross-sectional data such as observations). 
        The function can retain the original depth grid or rebin it to a specified vertical grid (using rebin_depth). If numpy_algo=True then the function will read in and return numpy arrays, otherwise it will use xarrays. 
        The cross-sections can be regular (e.g. along a single latitude index) or non-regular (e.g. along some arbitrary set of geo coordinates output by the SigmaZ_diag_ReadData function) in space. 
    
    Args:
        transp (NDarray): Cross-section of transport normal to the cross-section. 
        Rho (array): Cross-section of potential density (referenced to any pressure level)
        rholims (vector): a 2-element vector of the type [lowest_rho,highest_rho], that specifies the range of density-space density levels to calculate. 
        depth (1D array): the depth vector of the transp and rho cross-sections
        rebin_depth (1D array): Rebin the depth vector onto this vector. Helpful to convert an original depth vector with uneven grid spacing to a regular-spaced vertical grid. If left empty then it won't rebin. 
        nrho (integer): number of density bins in the range set by rholims. 
        z_layer_var (string): name of the z-coordinate (redundant if numpy_algo=True)
        dist_var (string): name of the horizontal coordinate (the default, 'sect', is the naming convection in Raf's sectionate tool). (redundant if numpy_algo=True)
        time_var (string): Name of the time dimension. 
        numpy_algo (logical): If True then the function takes and returns regular arrays, otherwise xarrays. 
        
    output:
        ty_z_rho (ND array): SigmaZ diagram on the original depth coordinates
        ty_z_rho_rebin (ND array): SigmaZ diagram on the rebin_depth coordinates (if rebin_depth was specified. Otherwise it is empty)
        rho_bounds (1D array): cell edges of the SigmaZ vertical density coordinate 
        rho_ref (1D array): cell centre of the SigmaZ vertical density coordinate 

    """

    rho_ref=rholims[0]+np.arange(0,nrho-1)*((rholims[1]-rholims[0])/nrho); 
    rho_bounds=rholims[0]-(rho_ref[1]-rho_ref[0])/2+np.arange(0,nrho)*((rholims[1]-rholims[0])/nrho); #the grid edges of the density vertical axis

    ### Rebin transports into SigmaZ space
    if numpy_algo:
        
        model_depth=depth[1:]  
        depth_diff=np.diff(depth)
        transp[np.isnan(transp)]=0
        rho[np.isnan(rho)]=0
        ty_z_rho=make_sigma_z(rho_bounds,transp,rho)
        ty_z_rho=ty_z_rho
        ty_z_rho[np.isnan(ty_z_rho)] = 0

    else:
        model_depth=depth.isel({ z_layer_var : slice(1,len(depth))})
        depth_diff=depth.diff(z_layer_var)
        ty_z_rho=histogram(rho,bins=[rho_bounds],weights=transp.fillna(0.),dim=[dist_var])
    
    ### If a rebin_depth is given, rebin the SigmaZ onto those depths instead: 
    if len(rebin_depth)>0:
        if numpy_algo:
            ty_z_rho_rebin=rebin_sigma_z(model_depth,depth_diff,rebin_depth,ty_z_rho)
        else:
            ty_z_rho_rebin=rebin_sigma_z(model_depth.values,depth_diff.values,rebin_depth,ty_z_rho.values)
            ty_z_rho_rebin=xr.DataArray(data=ty_z_rho_rebin, dims=(time_var,'z_l_rebin','rho_bin'), 
                                        coords={time_var : ty_z_rho[time_var],'z_l_rebin' : rebin_depth, 'rho_bin' : ty_z_rho['rho_bin']})
    else:
        ty_z_rho_rebin=[]
    
    return ty_z_rho, ty_z_rho_rebin, rho_bounds, rho_ref


def SigmaZ_diag_PlotDiag(ty_z_rho,depth,rho_ref,rho_bounds,rho0=1030,min_sigma_cutoff=0,min_z_cutoff=0,plot_flag=True,sig_axis_plot_lims=[26.5,28],z_axis_plot_lims=[100,4000],z_dim='z_l_rebin',rho_dim='rho_bin',numpy_algo=False):
    """ Makes some basic plots of the SigmaZ diagram and some quantities derived from it, such as overturning streamfunctions and timeseries in both depth- and density- space. These quantities are also output so that the user can make their own plots. 
    
    Args:
        ty_z_rho (ND array): SigmaZ array output by the SigmaZ_diag_ComputeDiag function. 
        depth (1D array): depth coordinate of ty_z_rho
        rho_ref (1D array): cell centre of the SigmaZ vertical density coordinate 
        rho_bounds (1D array): cell edges of the SigmaZ vertical density coordinate 
        rho0 (float): Reference pressure
        min_sigma_cutoff (float): when finding max(streamfunction), restricts density range to below this value
        min_z_cutoff (float): when finding max(streamfunction), restricts depth range to below this value
        plot_flag (logical): specifies whether to make a plot 
        sig_axis_plot_lims (vector): 2-element vector specifying max and min range of the density coordinate to plot
        z_axis_plot_lims (vector): 2-element vector specifying max and min range of the depth coordinate to plot
        z_dim (string): name of the depth coordinate in the SigmaZ array. Defunct if numpy_algo=True
        rho_dim (string): name of the density coordinate in the SigmaZ array. Defunct if numpy_algo=True
        numpy_algo=False (Logical): If True, the function takes and outputs standard arrays, otherwise it uses xarray. Ensure units are (time,z,sigma)
    
    output: 
        MOCz (array): overturning streamfunction in z-space 
        MOCrho (array): overturning streamfunction in sigma-space 
        ty_z (array): integrated transports along each in z-space level (i.e. MOCz is the cumulative sum along ty_z)
        ty_rho (array): integrated transports along each in sigma-space level (i.e. MOCrho is the cumulative sum along ty_rho)
        MOCz_ts (1D array): timeseries of the maximum MOCz streamfunction
        MOCrho_ts (1D array): timeseries of the maximum MOCrho streamfunction

    """
    
    if numpy_algo:
        ty_z = ty_z_rho.sum(axis=2)
        ty_rho = ty_z_rho.sum(axis=1)
        MOCz = np.cumsum(ty_z,axis=1)
        MOCrho = np.cumsum(ty_rho,axis=1)
        MOCrho_ts = np.max(MOCrho,axis=1)
        MOCz_ts = np.max(MOCz,axis=1)
    else:
        ty_z = ty_z_rho.sum(dim=rho_dim)
        ty_rho = ty_z_rho.sum(dim=z_dim)
        MOCz = ty_z.cumsum(dim=z_dim)
        MOCrho = ty_rho.cumsum(dim=rho_dim)
        MOCrho_ts = MOCrho.where(MOCrho[rho_dim]>min_sigma_cutoff).max(dim=rho_dim)
        MOCz_ts = MOCz.where(MOCz[z_dim]>min_z_cutoff).max(dim=z_dim)


    if plot_flag:
        fig = plt.figure(figsize=(12,7))
        ax = fig.add_subplot(2,2,3)
        ax.plot(MOCz.mean(axis=0)/rho0/1e6,-depth)
        ax.set_title('Xsection MOCz profile')
        ax.set_ylabel('z (m)')
        ax.set_xlabel('V (Sv)')
        plt.ylim([-z_axis_plot_lims[1],-z_axis_plot_lims[0]])
        ax = fig.add_subplot(2,2,1)
        ax.plot(MOCrho.mean(axis=0)/rho0/1e6,-rho_ref)
        ax.set_title('Xsection MOCsig profile')
        ax.set_ylabel('sigma (kg/m3)')
        plt.ylim([-sig_axis_plot_lims[1],-sig_axis_plot_lims[0]])
        ax = fig.add_subplot(2,2,4)
        ax.plot(MOCz_ts/rho0/1e6)
        ax.set_title('Xsection MOCz timeseries')
        ax.set_ylabel('V (Sv)')
        ax.set_xlabel('time')
        ax = fig.add_subplot(2,2,2)
        ax.plot(MOCrho_ts/rho0/1e6)
        ax.set_title('Xsection MOCsig timeseries')
        ax.set_ylabel('V (Sv)')
        

    
        spec = gridspec.GridSpec(ncols=2, nrows=2,
                         width_ratios=[1,2], wspace=0.3,
                         hspace=0.3, height_ratios=[1, 2])
        fig = plt.figure(figsize=(10,8))
        ax0 = fig.add_subplot(spec[3])
        ch1=ax0.pcolormesh(rho_bounds[0:-1],-depth,ty_z_rho.mean(axis=0)/rho0/1e6,vmin=-5e-1,vmax=5e-1,cmap='RdBu_r')
        ax0.set_title('Xsec transp. in Sigma-Z')
        ax0.set_xlabel('Sigma (kg/m3)')
        plt.xlim( [sig_axis_plot_lims[0],sig_axis_plot_lims[1]] ), plt.ylim( [-z_axis_plot_lims[1],-z_axis_plot_lims[0]] )
        box = ax0.get_position()
        ax0.set_position([box.x0*1, box.y0, box.width, box.height])
        axColor = plt.axes([box.x0*1 + box.width * 1.05, box.y0, 0.01, box.height])
        plt.colorbar(ch1, cax = axColor, orientation="vertical")
        #plt.show()
        ax1 = fig.add_subplot(spec[1])
        ax1.plot(rho_ref,ty_rho.mean(axis=0)/rho0/1e6)
        ax1.set_title('Xsec transp. in Sigma')
        ax1.set_ylabel('V (Sv)')
        plt.xlim([sig_axis_plot_lims[0],sig_axis_plot_lims[1]]), plt.ylim( [-2,2] )
        ax2 = fig.add_subplot(spec[2])
        ax2.plot(ty_z.mean(axis=0)/rho0/1e6,-depth)
        ax2.set_title('Xsec transp. in Z')
        ax2.set_xlabel('V (Sv)')
        ax2.set_ylabel('z (m)')    
        plt.ylim([-z_axis_plot_lims[1],-sig_axis_plot_lims[0]]), plt.xlim([-2,2]),
 
    return MOCz, MOCrho, ty_z, ty_rho, MOCz_ts, MOCrho_ts

def SigmaZ_diag_computeMFTMHT(transp,so,thetao,rho,cell_area,rho_bounds,depth,rebin_depth=[],rho_dim='rho_bin',z_dim='z_l',dist_var='sect',time_var='time',rho0=1030,annual_mean_flag=False,plot_flag=True):
    """ Calculates Heat and Freshwater transports normal to the cross-section, using input z-space cross-sections of transport, salinity, temperature and density. SigmaZ arrays of each of the properties are created so that heat and freshwater transports can be output in both depth- and density- coordinates. "Gyre" and "AMOC" components (in both depth- and density- space) of MHT and MFT are also calculated. This routine accepts and returns xarrays, but will still work if standard (numpy) arrays are input (in this case make sure dimension order is (time,z,x)) though it will convert everything to xarray. 
    
    Args:
        transp (ND array): z-space cross-section of meridional transport (normal to the cross-section). 
        so (ND array): z-space cross-section of salinity. 
        thetao (ND array): z-space cross-section of temperature. 
        rho (ND array): z-space cross-section of potential density. 
        cell_area (2D xarray): cross-sectional cell area
        rho_bounds (1D array): cell edges of the SigmaZ vertical density coordinate 
        depth (1D array): depth coordinate of the cross-section
        rebin_depth (1D array): Rebin the depth vector onto this vector. Helpful to convert an original depth vector with uneven grid spacing to a regular-spaced vertical grid. If left empty then it won't rebin. 
        rho_dim (string): name of the density coordinate. 
        z_dim (string): name of the depth coordinate. 
        dist_var (string): name of the horizontal coordinate. 
        time_var (string): name of the time coordinate. 
        annual_mean_flag (logical): If True then output will be converted from monthly to annual means. Must be set to False if not using monthly data
        plot_flag (logical): makes a simple plot of the output timeseries of MHT, MFT, and their overturning and gyre components. 
        
    output: 
         MOCsig_MHTMFT (xarray Dataset): dataset containing all density-space variables. Descriptions are provided in the output metadata        
         MOCz_MHTMFT (xarray Dataset): dataset containing all depth-space variables. Descriptions are provided in the output metadata
         MOCsigz_MHTMFT (xarray Dataset): dataset containing the SigmaZ diagrams of transport-weighted temperature and salinity. Descriptions in the metadata
        
    """
    
    #######################
    ### If numpy arrays are given, convert everything to xarray:
    
    if type(transp)==np.ndarray:
        transp=xr.DataArray(data=transp,dims=(time_var,z_dim,dist_var)).rename('uvnormal')
        so=xr.DataArray(data=so,dims=(time_var,z_dim,dist_var)).rename('so')
        thetao=xr.DataArray(data=thetao,dims=(time_var,z_dim,dist_var)).rename('thetao')
        rho=xr.DataArray(data=rho,dims=(time_var,z_dim,dist_var)).rename('rho')
        cell_area=xr.DataArray(data=cell_area,dims=(time_var,z_dim,dist_var)).rename('section_cellarea')
        depth=xr.DataArray(data=depth,dims=(z_dim)).rename('cell_depth')
        
    ######################
    ### calculate SigmaZ matrices for T,S and Rho:

    Cp=3850 # heat capacity of seawater      

    ty_z_rho = histogram(rho,bins=[rho_bounds],weights=transp.fillna(0.),dim=[dist_var])
    thetao_z_rho = histogram(rho,bins=[rho_bounds],weights=(thetao*cell_area).fillna(0.),dim=[dist_var])
    so_z_rho = histogram(rho,bins=[rho_bounds],weights=(so*cell_area).fillna(0.),dim=[dist_var])
    cellarea_z_rho = histogram(rho,bins=[rho_bounds],weights=(cell_area).fillna(0.),dim=[dist_var])
    thetao_z_rho_mean = thetao_z_rho/cellarea_z_rho  # The mean temperature in each sigma_z cell for OSNAP 
    so_z_rho_mean = so_z_rho/cellarea_z_rho   # The mean salinity in each sigma_z cell for OSNAP 
    
    ######################
    ### if rebin_depth is specified, remap the depth coordinate from the native coordinate to the specified one.  
    if len(rebin_depth)>0:
        model_depth=depth.isel({ z_dim : slice(1,len(depth))})
        depth_diff=depth.diff(z_dim)
        thetao_z_rho=rebin_sigma_z(model_depth.values,depth_diff.values,rebin_depth,thetao_z_rho.values)
        thetao_z_rho=xr.DataArray(data=thetao_z_rho, dims=(time_var,z_dim,rho_dim),coords={time_var : ty_z_rho[time_var],z_dim : rebin_depth, rho_dim : ty_z_rho[rho_dim]})
        so_z_rho=rebin_sigma_z(model_depth.values,depth_diff.values,rebin_depth,so_z_rho.values)
        so_z_rho=xr.DataArray(data=so_z_rho, dims=(time_var,z_dim,rho_dim),coords={time_var : ty_z_rho[time_var],z_dim : rebin_depth, rho_dim : ty_z_rho[rho_dim]})
        cellarea_z_rho=rebin_sigma_z(model_depth.values,depth_diff.values,rebin_depth,cellarea_z_rho.values)
        cellarea_z_rho=xr.DataArray(data=cellarea_z_rho, dims=(time_var,z_dim,rho_dim),coords={time_var : ty_z_rho[time_var],z_dim : rebin_depth, rho_dim : ty_z_rho[rho_dim]})
        ty_z_rho=rebin_sigma_z(model_depth.values,depth_diff.values,rebin_depth,ty_z_rho.values)
        ty_z_rho=xr.DataArray(data=ty_z_rho, dims=(time_var,z_dim,rho_dim),coords={time_var : thetao_z_rho[time_var],z_dim : rebin_depth, rho_dim : thetao_z_rho[rho_dim]})

    #######################
    #### Calculate z- and rho- space zonal means of the T,S and V SigmaZ diagrams, for use in calculating overturning component of MFT and MHT:
    
    S_bar=so_z_rho.fillna(0).sum()/cellarea_z_rho.fillna(0).sum() 
    S_zm_z = so_z_rho.fillna(0).sum(dim=rho_dim)/cellarea_z_rho.fillna(0).sum(dim=rho_dim) 
    S_zm_rho = so_z_rho.fillna(0).sum(dim=z_dim)/cellarea_z_rho.fillna(0).sum(dim=z_dim) # rho-space zonal mean salinity
    theta_zm_z = thetao_z_rho.fillna(0).sum(dim=rho_dim)/cellarea_z_rho.fillna(0).sum(dim=rho_dim) 
    theta_zm_rho = thetao_z_rho.fillna(0).sum(dim=z_dim)/cellarea_z_rho.fillna(0).sum(dim=z_dim) 
    ty_zm_z = ty_z_rho.fillna(0).sum(dim=rho_dim)
    ty_zm_rho = ty_z_rho.fillna(0).sum(dim=z_dim)

    #######################
    ### calculate MOC component of MFT and MHT:
    
    MFT_MOCrho =- ty_zm_rho*(S_zm_rho-S_bar)/S_bar
    MFT_MOCz =- ty_zm_z*(S_zm_z-S_bar)/S_bar
    MFT_MOCrho_sum = MFT_MOCrho.fillna(0).sum(dim=rho_dim)
    MFT_MOCz_sum = MFT_MOCz.fillna(0).sum(dim=z_dim)
    MHT_MOCrho = Cp*ty_zm_rho*theta_zm_rho
    MHT_MOCz = Cp*ty_zm_z*theta_zm_z
    MHT_MOCrho_sum = MHT_MOCrho.fillna(0).sum(dim=rho_dim)
    MHT_MOCz_sum = MHT_MOCz.fillna(0).sum(dim=z_dim)

    ########################
    ### calculate the transport*tracer SigmaZ matrices, and calculate (zonal- and total-) integrated MFT  and MHT:
    
    Vthetao_z_rho = histogram(rho,bins=[rho_bounds],weights=(thetao*transp).fillna(0.),dim=[dist_var])
    Vso_z_rho = -histogram(rho,bins=[rho_bounds],weights=((so-S_bar)*transp/S_bar).fillna(0.),dim=[dist_var])
    
    ######################
    ### if rebin_depth is specified, remap the transport*tracer SigmaZ matrices from the native vertical coordinate to the specified one: 

    if len(rebin_depth)>0:
        Vthetao_z_rho=rebin_sigma_z(model_depth.values,depth_diff.values,rebin_depth,Vthetao_z_rho.values)
        Vthetao_z_rho=xr.DataArray(data=Vthetao_z_rho, dims=(time_var,z_dim,rho_dim),coords={time_var : ty_z_rho[time_var],z_dim : rebin_depth, rho_dim : ty_z_rho[rho_dim]})
        Vso_z_rho=rebin_sigma_z(model_depth.values,depth_diff.values,rebin_depth,Vso_z_rho.values)
        Vso_z_rho=xr.DataArray(data=Vso_z_rho, dims=(time_var,z_dim,rho_dim),coords={time_var : ty_z_rho[time_var],z_dim : rebin_depth, rho_dim : ty_z_rho[rho_dim]})

    ######################
    ### calculate the GYRE component of MFT and MHT:
    
    MFT_zonmean_rho=Vso_z_rho.fillna(0).sum(dim=z_dim)
    MFT_zonmean_z=Vso_z_rho.fillna(0).sum(dim=rho_dim)
    MFT_GYRErho=MFT_zonmean_rho-MFT_MOCrho
    MFT_GYREz=MFT_zonmean_z-MFT_MOCz
    MFT_GYRErho_sum=MFT_GYRErho.fillna(0).sum(dim=rho_dim)
    MFT_GYREz_sum=MFT_GYREz.fillna(0).sum(dim=z_dim)
    MFT_sum=Vso_z_rho.fillna(0.).sum(dim=[z_dim,rho_dim])
    MHT_zonmean_rho=Cp*Vthetao_z_rho.fillna(0).sum(dim=z_dim)
    MHT_zonmean_z=Cp*Vthetao_z_rho.fillna(0).sum(dim=rho_dim)
    MHT_GYRErho=MHT_zonmean_rho-MHT_MOCrho
    MHT_GYREz=MHT_zonmean_z-MHT_MOCz
    MHT_GYRErho_sum=MHT_GYRErho.fillna(0).sum(dim=rho_dim)
    MHT_GYREz_sum=MHT_GYREz.fillna(0).sum(dim=z_dim)
    MHT_sum=Cp*Vthetao_z_rho.fillna(0.).sum(dim=[z_dim,rho_dim])

    #######################
    ### Calculate depth- and density- space overturning:
    
    MOCrho =ty_z_rho.sum(dim=z_dim).cumsum(dim=rho_dim).max(dim=rho_dim)
    MOCz = ty_z_rho.sum(dim=rho_dim).cumsum(dim=z_dim).max(dim=z_dim)
    
    #######################
    ### For convenience, combine the output into individual datasets for sigma-, z-, and SigmaZ-space:
    
    MOCsig_MHTMFT = xr.Dataset()
    MOCsig_MHTMFT['MOCrho']=MOCrho;    
    MOCsig_MHTMFT['MOCrho'].attrs['long_name'] = 'maximum overturning in density-space'    
    MOCsig_MHTMFT['MHT_sum']=MHT_sum;    
    MOCsig_MHTMFT['MHT_sum'].attrs['long_name'] = 'Sum density-space Meridional Heat Transport across section'
    MOCsig_MHTMFT['MHT_zonmean_rho']=MHT_zonmean_rho;    
    MOCsig_MHTMFT['MHT_zonmean_rho'].attrs['long_name'] = 'density-space Meridional Heat Transport averaged along density-layers'
    MOCsig_MHTMFT['MHT_MOCrho']=MHT_MOCrho;    
    MOCsig_MHTMFT['MHT_MOCrho'].attrs['long_name'] = 'density-space MOC component of MHT integrated along each density layer'
    MOCsig_MHTMFT['MHT_MOCrho_sum']=MHT_MOCrho_sum;    
    MOCsig_MHTMFT['MHT_MOCrho_sum'].attrs['long_name'] = 'Sum of the density-space MOC component of MHT'
    MOCsig_MHTMFT['MHT_GYRErho']=MHT_GYRErho;    
    MOCsig_MHTMFT['MHT_GYRErho'].attrs['long_name'] = 'density-space Gyre component of MHT integrated along each density layer'
    MOCsig_MHTMFT['MHT_GYRErho_sum']=MHT_GYRErho_sum;    
    MOCsig_MHTMFT['MHT_GYRErho_sum'].attrs['long_name'] = 'Sum of the density-space GYRE component of MHT'
    MOCsig_MHTMFT['MFT_sum']=MFT_sum;    
    MOCsig_MHTMFT['MFT_sum'].attrs['long_name'] = 'Sum density-space Meridional Freshwater Transport across section'
    MOCsig_MHTMFT['MFT_zonmean_rho']=MFT_zonmean_rho;    
    MOCsig_MHTMFT['MFT_zonmean_rho'].attrs['long_name'] = 'density-space Meridional Freshwater Transport averaged along density-layers'
    MOCsig_MHTMFT['MFT_MOCrho']=MFT_MOCrho;    
    MOCsig_MHTMFT['MFT_MOCrho'].attrs['long_name'] = 'MOC component of MFT integrated along each density layer'
    MOCsig_MHTMFT['MFT_MOCrho_sum']=MFT_MOCrho_sum;   
    MOCsig_MHTMFT['MFT_MOCrho_sum'].attrs['long_name'] = 'Sum of the density-space MOC component of MFT'
    MOCsig_MHTMFT['MFT_GYRErho']=MFT_GYRErho;    
    MOCsig_MHTMFT['MFT_GYRErho'].attrs['long_name'] = 'density-space Gyre component of MFT integrated along each density layer'
    MOCsig_MHTMFT['MFT_GYRErho_sum']=MFT_GYRErho_sum;   
    MOCsig_MHTMFT['MFT_GYRErho_sum'].attrs['long_name'] = 'Sum of the density-space GYRE component of MFT'

    MOCz_MHTMFT = xr.Dataset()
    MOCz_MHTMFT['MOCz']=MOCz
    MOCz_MHTMFT['MOCz'].attrs['long_name'] = 'maximum overturning in depth-space'
    MOCz_MHTMFT['MHT_sum']=MHT_sum
    MOCz_MHTMFT['MHT_sum'].attrs['long_name'] = 'Sum depth-space Meridional Heat Transport across section'
    MOCz_MHTMFT['MHT_zonmean_z']=MHT_zonmean_z;    
    MOCz_MHTMFT['MHT_zonmean_z'].attrs['long_name'] = 'depth-space Meridional Heat Transport averaged along depth-layers'
    MOCz_MHTMFT['MHT_MOCz']=MHT_MOCz
    MOCz_MHTMFT['MHT_MOCz'].attrs['long_name'] = 'depth-space MOC component of MHT integrated along each depth layer'
    MOCz_MHTMFT['MHT_MOCz_sum']=MHT_MOCz_sum
    MOCz_MHTMFT['MHT_MOCz_sum'].attrs['long_name'] = 'Sum of the depth-space MOC component of MHT'
    MOCz_MHTMFT['MHT_GYREz']=MHT_GYREz
    MOCz_MHTMFT['MHT_GYREz'].attrs['long_name'] = 'depth-space Gyre component of MHT integrated along each depth layer'
    MOCz_MHTMFT['MHT_GYREz_sum']=MHT_GYREz_sum
    MOCz_MHTMFT['MHT_GYREz_sum'].attrs['long_name'] = 'Sum of the depth-space GYRE component of MHT'
    MOCz_MHTMFT['MFT_sum']=MFT_sum
    MOCz_MHTMFT['MFT_sum'].attrs['long_name'] = 'Sum depth-space Meridional Freshwater Transport across section'
    MOCz_MHTMFT['MFT_zonmean_z']=MFT_zonmean_z;    
    MOCz_MHTMFT['MFT_zonmean_z'].attrs['long_name'] = 'depth-space Meridional Freshwater Transport averaged along depth-layers'
    MOCz_MHTMFT['MFT_MOCz']=MFT_MOCz
    MOCz_MHTMFT['MFT_MOCz'].attrs['long_name'] = 'MOC component of MFT integrated along each depth layer'
    MOCz_MHTMFT['MFT_MOCz_sum']=MFT_MOCz_sum
    MOCz_MHTMFT['MFT_MOCz_sum'].attrs['long_name'] = 'Sum of the depth-space MOC component of MFT'
    MOCz_MHTMFT['MFT_GYREz']=MFT_GYREz
    MOCz_MHTMFT['MFT_GYREz'].attrs['long_name'] = 'depth-space Gyre component of MFT integrated along each depth layer'
    MOCz_MHTMFT['MFT_GYREz_sum']=MFT_GYREz_sum
    MOCz_MHTMFT['MFT_GYREz_sum'].attrs['long_name'] = 'Sum of the depth-space GYRE component of MFT'

    MOCsigz_MHTMFT = xr.Dataset()
    MOCsigz_MHTMFT['Vthetao_z_rho']=Vthetao_z_rho
    MOCsigz_MHTMFT['Vthetao_z_rho'].attrs['long_name'] = 'SigmaZ diagram of Transport-weighted temperature. Integrate (and multiple by Cp) to get total MHT'
    MOCsigz_MHTMFT['Vso_z_rho']=Vso_z_rho
    MOCsigz_MHTMFT['Vso_z_rho'].attrs['long_name'] = 'SigmaZ diagram of Transport-weighted salinity. Integrate to get total MFT'
    
    ########################
    ### if annual_mean_flag=True, temporally rebin the monthly offline timeseries into annual ones. Used in the plot below:
    
    if annual_mean_flag:
        MOCsig_MHTMFT=MOCsig_MHTMFT.coarsen(time=12).mean()
        MOCz_MHTMFT=MOCz_MHTMFT.coarsen(time=12).mean()
        MOCsigz_MHTMFT=MOCsigz_MHTMFT.coarsen(time=12).mean()
    
    ########################
    ### Plot some timeseries (if plot_flag is set to True):
    
    if plot_flag:
        fig = plt.figure(figsize=(10,12))
        ax = fig.add_subplot(3,1,1)
        # ax.plot(MOCsig_MHTMFT.MOCrho.time,MOCsig_MHTMFT.MOCrho/rho0/1e6)
        (MOCsig_MHTMFT.MOCrho/rho0/1e6).plot()
        ax.set_title('MOC in rho-space')
        ax.set_ylabel('Sv')
        ax = fig.add_subplot(3,1,2)
        # ax.plot(MOCsig_MHTMFT.MOCrho.time,(MOCsig_MHTMFT.MHT_sum)/1e15)
        # ax.plot(MOCsig_MHTMFT.MOCrho.time,(MOCsig_MHTMFT.MHT_MOCrho_sum)/1e15)
        # ax.plot(MOCsig_MHTMFT.MOCrho.time,(MOCsig_MHTMFT.MHT_GYRErho_sum)/1e15)
        (MOCsig_MHTMFT.MHT_sum/1e15).plot()
        (MOCsig_MHTMFT.MHT_MOCrho_sum/1e15).plot()
        (MOCsig_MHTMFT.MHT_GYRErho_sum/1e15).plot()
        ax.set_title('MHT in rho-space')
        ax.set_ylabel('PW')
        ax = fig.add_subplot(3,1,3)
        # ax.plot(MOCsig_MHTMFT.MOCrho.time,(MOCsig_MHTMFT.MFT_sum)/rho0/1e6,label='TOTrho')
        # ax.plot(MOCsig_MHTMFT.MOCrho.time,(MOCsig_MHTMFT.MFT_MOCrho_sum)/rho0/1e6,label='MOCrho')
        # ax.plot(MOCsig_MHTMFT.MOCrho.time,(MOCsig_MHTMFT.MFT_GYRErho_sum)/rho0/1e6,label='GYRErho')
        (MOCsig_MHTMFT.MFT_sum/rho0/1e6).plot(label='TOTrho')
        (MOCsig_MHTMFT.MFT_MOCrho_sum/rho0/1e6).plot(label='MOCrho')
        (MOCsig_MHTMFT.MFT_GYRErho_sum/rho0/1e6).plot(label='GYRErho')
        ax.legend(loc="lower left")
        ax.set_title('MFT in rho-space')
        ax.set_ylabel('Sv')
        ax.set_xlabel('time')

        
            
    return MOCsig_MHTMFT, MOCz_MHTMFT, MOCsigz_MHTMFT


@njit
def make_sigma_z(rho_bounds,transp_vals,rho_vals):
    """ Numpy version of calculating a SigmaZ diagram. The xarray version uses xhistogram instead (see above)""" 
    ty_z_rho=np.zeros((np.shape(transp_vals)[0],np.shape(transp_vals)[1],np.shape(rho_bounds)[0]-1));
    for k in range(0,np.shape(transp_vals)[0]):
        for krho in range(0,np.shape(rho_bounds)[0]-1):
           for kk in range(0,np.shape(transp_vals)[1]): 
               for ii in range(0,np.shape(transp_vals)[2]): 
                   if (rho_vals[k,kk,ii] >= rho_bounds[krho]) & (rho_vals[k,kk,ii]<rho_bounds[krho+1]):
                       ty_z_rho[k,kk,krho]=ty_z_rho[k,kk,krho]+transp_vals[k,kk,ii]
    return ty_z_rho


@njit
def rebin_sigma_z(model_depth,depth_diff,rebin_depth,ty_z_rho):
    """ Code to remap a SigmaZ diagram (ty_z_rho) from the native vertical grid (model depth) to a specified grid (rebin_depth)""" 
    ty_z_rho_rebin=np.zeros((np.shape(ty_z_rho)[0],np.shape(rebin_depth)[0],np.shape(ty_z_rho)[2]))
    rebin_depth_index=0;
    for ii in range(0,np.shape(ty_z_rho)[1]-1):
        V=ty_z_rho[:,ii,:]
        depth_range=depth_diff[ii]
        if model_depth[ii]<rebin_depth[rebin_depth_index]:
            ty_z_rho_rebin[:,rebin_depth_index,:]=ty_z_rho_rebin[:,rebin_depth_index,:]+V
        elif model_depth[ii]==rebin_depth[rebin_depth_index]:
            ty_z_rho_rebin[:,rebin_depth_index,:]=ty_z_rho_rebin[:,rebin_depth_index,:]+V
            rebin_depth_index=rebin_depth_index+1
        elif model_depth[ii]>rebin_depth[rebin_depth_index]:
            top_frac=V*(rebin_depth[rebin_depth_index]-model_depth[ii-1])/depth_range
            ty_z_rho_rebin[:,rebin_depth_index,:]=ty_z_rho_rebin[:,rebin_depth_index,:]+top_frac
            if model_depth[ii]<rebin_depth[rebin_depth_index+1]:
                rebin_depth_index=rebin_depth_index+1
                ty_z_rho_rebin[:,rebin_depth_index,:]=ty_z_rho_rebin[:,rebin_depth_index,:]+(V-top_frac)
            else:
                jjj=0
                while model_depth[ii]>rebin_depth[rebin_depth_index+1]:
                    rebin_depth_index=rebin_depth_index+1
                    middle_frac=V*(rebin_depth[rebin_depth_index]-rebin_depth[rebin_depth_index-1])/depth_range
                    ty_z_rho_rebin[:,rebin_depth_index,:]=ty_z_rho_rebin[:,rebin_depth_index,:]+middle_frac
                    jjj=jjj+1
                rebin_depth_index=rebin_depth_index+1
                ty_z_rho_rebin[:,rebin_depth_index,:]=ty_z_rho_rebin[:,rebin_depth_index,:]+(V-top_frac-(jjj*middle_frac))
    return ty_z_rho_rebin

def use_supergrid(grid,x_hpoint_1Dvar,x_qpoint_1Dvar,y_hpoint_1Dvar,y_qpoint_1Dvar):
    
    c_xindices=np.arange(1,grid.x.shape[1],2)
    c_yindices=np.arange(1,grid.y.shape[0],2)
    xindices=np.arange(0,grid.x.shape[1]-1,2)
    yindices=np.arange(0,grid.y.shape[0]-1,2)

    # c_xindices=np.arange(0,grid.x.shape[1],2)
    # c_yindices=np.arange(0,grid.y.shape[0],2)
    # xindices=np.arange(1,grid.x.shape[1]-1,2)
    # yindices=np.arange(1,grid.y.shape[0]-1,2)

    geolon_c=grid.x.isel(nyp=c_yindices,nxp=c_xindices)
    geolon=grid.x.isel(nyp=yindices,nxp=xindices)
    geolat_c=grid.y.isel(nyp=c_yindices,nxp=c_xindices)
    geolat=grid.y.isel(nyp=yindices,nxp=xindices)

    xq=geolon_c.isel(nyp=round(len(geolon_c.nyp)/2))
    yq=geolat_c.mean('nxp')
    xh=geolon.isel(nyp=round(len(geolon.nyp)/2))
    yh=geolat.mean('nxp')

    geolon_c['nxp']=xq; geolon_c['nyp']=yq
    geolat_c['nxp']=xq; geolat_c['nyp']=yq
    geolon['nxp']=xh; geolon['nyp']=yh
    geolat['nxp']=xh; geolat['nxp']=yh

    newgrid=xr.Dataset()
    newgrid['geolon_c']=geolon_c.rename({'nxp' : x_qpoint_1Dvar, 'nyp' : y_qpoint_1Dvar})
    newgrid['geolat_c']=geolat_c.rename({'nxp' : x_qpoint_1Dvar, 'nyp' : y_qpoint_1Dvar})
    newgrid['geolon']=geolon.rename({'nxp' : x_hpoint_1Dvar, 'nyp' : y_hpoint_1Dvar})
    newgrid['geolat']=geolat.rename({'nxp' : x_hpoint_1Dvar, 'nyp' : y_hpoint_1Dvar})
    
    grid=newgrid
    
    return grid

def calc_partial_cells(dir_deptho,ds,grid,section_node_lons,section_node_lats,x_hpoint_1Dvar,x_qpoint_1Dvar,y_hpoint_1Dvar,y_qpoint_1Dvar,u_transport_var,v_transport_var):

    deptho=xr.open_dataset(dir_deptho).deptho

    if y_hpoint_1Dvar != "yh":
        deptho=deptho.rename({y_hpoint_1Dvar: 'yh'})
    if x_hpoint_1Dvar != "xh":
        deptho=deptho.rename({x_hpoint_1Dvar: 'xh'})

    deptho['xh']=ds['xh']
    deptho['yh']=ds['yh']
    
    ### Reduce xarray data domain to fit around chosen section coordinates
    lat_range_min=np.abs(ds.yh-(min(section_node_lats)-10)).argmin()
    lat_range_max=np.abs(ds.yh-(max(section_node_lats)+10)).argmin()
    lon_range_min=np.abs(ds.xh-(min(section_node_lons)-10)).argmin()
    lon_range_max=np.abs(ds.xh-(max(section_node_lons)+10)).argmin()
    ds_subpolar = ds.sel(yq=slice(ds.yq[lat_range_min],ds.yq[lat_range_max]),
                         xh=slice(ds.xh[lon_range_min],ds.xh[lon_range_max]),
                         yh=slice(ds.yh[lat_range_min],ds.yh[lat_range_max]),
                         xq=slice(ds.xq[lon_range_min],ds.xq[lon_range_max]))
    grid_region = grid.sel(yq=slice(grid.yq[lat_range_min],grid.yq[lat_range_max]),
                             xh=slice(grid.xh[lon_range_min],grid.xh[lon_range_max]),
                             yh=slice(grid.yh[lat_range_min],grid.yh[lat_range_max]),
                             xq=slice(grid.xq[lon_range_min],grid.xq[lon_range_max]))
    deptho_region = deptho.sel(xh=slice(deptho.xh[lon_range_min],deptho.xh[lon_range_max]),
                               yh=slice(deptho.yh[lat_range_min],deptho.yh[lat_range_max]))
    
    deptho_u=((deptho_region.isel(xh=slice(0,len(deptho_region.xh-1)))+deptho_region.isel(xh=slice(1,len(deptho_region.xh))))/2).pad(xh=(1,0)).rename({'xh' : 'xq'})
    deptho_v=((deptho_region.isel(yh=slice(0,len(deptho_region.yh-1)))+deptho_region.isel(yh=slice(1,len(deptho_region.yh))))/2).pad(yh=(1,0)).rename({'yh' : 'yq'})
    deptho_u['xq']=ds_subpolar['xq']
    deptho_v['yq']=ds_subpolar['yq']
    
    z_i_u=ds.z_i.isel(z_i=slice(1,len(ds.z_i))).rename({'z_i' : 'z_l'}).expand_dims(dim={'xq' : deptho_u.xq, 'yh' : deptho_u.yh})
    z_i_v=ds.z_i.isel(z_i=slice(1,len(ds.z_i))).rename({'z_i' : 'z_l'}).expand_dims(dim={'xh' : deptho_v.xh, 'yq' : deptho_v.yq})
    dz_i_u=ds.z_i.diff(dim='z_i').rename({'z_i' : 'z_l'}).expand_dims(dim={'xq' : deptho_u.xq, 'yh' : deptho_u.yh})
    dz_i_v=ds.z_i.diff(dim='z_i').rename({'z_i' : 'z_l'}).expand_dims(dim={'xh' : deptho_v.xh, 'yq' : deptho_v.yq})
    z_i_u['z_l']=ds['z_l']
    z_i_v['z_l']=ds['z_l']
    dz_i_u['z_l']=ds['z_l']
    dz_i_v['z_l']=ds['z_l']    
    
    apparent_depth_u=z_i_u.where(ds_subpolar[u_transport_var].isel(time=0).notnull())
    apparent_depth_v=z_i_v.where(ds_subpolar[v_transport_var].isel(time=0).notnull())
    
    depth_discrepancy_u=deptho_u-apparent_depth_u.max(dim='z_l')
    depth_discrepancy_v=deptho_v-apparent_depth_v.max(dim='z_l')
    depth_discrepancy_u_grid=apparent_depth_u-(deptho_u.expand_dims(dim={'z_l' : ds.z_l}))
    depth_discrepancy_v_grid=apparent_depth_v-(deptho_v.expand_dims(dim={'z_l' : ds.z_l}))
    
    partial_cell_u = depth_discrepancy_u_grid.where(apparent_depth_u==apparent_depth_u.max(dim='z_l'))/dz_i_u.where(apparent_depth_u==apparent_depth_u.max(dim='z_l'))
    partial_cell_u=partial_cell_u.where(partial_cell_u.notnull(),1)
    partial_cell_u=partial_cell_u.where(partial_cell_u<1,1)
    
    partial_cell_v = depth_discrepancy_v_grid.where(apparent_depth_v==apparent_depth_v.max(dim='z_l'))/dz_i_v.where(apparent_depth_v==apparent_depth_v.max(dim='z_l'))
    partial_cell_v=partial_cell_v.where(partial_cell_v.notnull(),1)
    partial_cell_v=partial_cell_v.where(partial_cell_v<1,1)

    ds_partial_cell=xr.Dataset()
    ds_partial_cell['partial_cell_v']=partial_cell_v
    ds_partial_cell['partial_cell_u']=partial_cell_u
    ds_partial_cell['z_i']=ds['z_i']
    
    ### Run Raf's sectionate tool to extract T,S and V along chosen section coordinates
    isec, jsec, xsec, ysec = sectionate.create_section_composite(grid_region['geolon_c'],
                                                                 grid_region['geolat_c'],
                                                                 section_node_lons,
                                                                 section_node_lats)    
    
    partial_cell_section = sectionate.sectionate_gridwidth(ds_partial_cell,ds_subpolar,isec,jsec,uname='partial_cell_u',vname='partial_cell_v')
    
    return partial_cell_section