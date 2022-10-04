import matplotlib.pyplot as plt
from matplotlib import gridspec
import om4labs
import xarray as xr
import numpy as np
import sectionate
from glob import glob
import momlevel
from numba import njit
from xhistogram.xarray import histogram


def SigmaZ_diag_ReadData(
    dir_base,
    section_node_lons,
    section_node_lats,
    file_str_identifier,
    calc_rho_flag=True,
    dir_vars="gfdl.ncrc4-intel18-prod-openmp/pp/ocean_month_z/ts/monthly/5yr/",
    dir_grid="gfdl.ncrc4-intel18-prod-openmp/pp/ocean_month_z/ocean_month_z.static.nc",
    z_layer_var="z_l",
    z_inter_var="z_i",
    u_transport_var="umo",
    v_transport_var="vmo",
    theta_var="thetao",
    salt_var="so",
    rho_var="rhopot0",
    time_limits=[],
    ref_pres=0,
    x_hpoint_1Dvar="xh",
    x_qpoint_1Dvar="xq",
    y_hpoint_1Dvar="yh",
    y_qpoint_1Dvar="yq",
    time_var="time",
    lons_tpoint="geolon",
    lats_tpoint="geolat",
    lons_cpoint="geolon_c",
    lats_cpoint="geolat_c",
    decode_times_flag=True,
    mom_gridwidth_algo=True,
):
    """Read model data to be used for calculating a SigmaZ diagram (See Zhang and Thomas, 2021, Nat Comm Earth Env.) along a specified cross section. 
    
    Args:
        dir_base (string): base directory of the simulation to be used e.g. "/archive/Raphael.Dussin/FMS2019.01.03_devgfdl_20210706/CM4_piControl_c192_OM4p125_v6_alt3/"
        section_node_lons (vector): 1D vector of longitude nodes that (together with section_node_lats) define a cross section. E.g.: [60.3000, 58.8600, 58.0500, 58.0000, 56.5000]
        section_node_lats (vector): 1D vector of latitude nodes that define the cross section. For use in Raphael Dussin's sectionate tool, which finds all grid cell locations along straight lines between each lon/lat node. E.g [-44.9000, -30.5400, -28.0000, -14.7000, -5.9300]
        file_str_identifier (string): string that identifies a particular output file or set of output files. E.g. ".0196*", for all strings containing '.0196', or *.0[1-2]* for years 100-200
        calc_rho_flag (logical): If calc_rho_flag==True, calculate sigma0 from T and S files (according to variables "theta_var" and "salt_var"). Otherwise read sigma0 directly from file according to variable name "rho_var". 
        dir_vars (string): subdirectory containing the depth-space output files. 
        dir_grid (string): subdirectory cotaining the grid info file
        z_layer_var (string): variable name for the depth coordinate at the grid centre
        z_inter_var (string): variable name for the depth coordinate at the grid interface
        u_transport_var (string): variable name for the zonal transport
        v_transport_var (string): variable name for the meridional transport
        theta_var (string): variable name for the potential temperature (if being read, as per the calc_rho_flag)
        salt_var (string): variable name for salt (if being read, as per the calc_rho_flag)
        rho_var (string): variable name for rho (if being read, as per the calc_rho_flag)
        time_limits (vector): a 2-element vector of the type [time_start,time_limit], using the same units as in the data, that allows the data to be limited within these time margins. If using decode_times=True then the vector can contain year strings  (e.g. time_limits=['2014','2020']), otherwise it can also contain numbers (e.g. time_limits=[8000,10000])
        ref_pres (scalar): Reference pressure in Pa (e.g. 2000 m is approx 20000 Pa)
        x_hpoint_1Dvar (string) etc: coordinate names of the 1D lon and lat q and h points
        lons_tpoint (string) etc: coordinate names for the 2D lon and lat grid at the grid centre
        lons_cpoint (string) etc: coordinate names for the 2D lon and lat grid at the grid corner
        time_var (string): coordinate name of the time dimension
        decode_times_flag (logical): use metadata to convert time to a date

    output:
        dsT (xarray dataset):  output from Raphael Dussin's sectionate tool, containing transport and grid information along the specified cross-section
        transp (numpy array): cross section of transports perpendicular to the specified cross section
        rho(numpy array): cross section of density along the specified cross section

    """
    
    umo_vars_str=dir_base+dir_vars+"*"+file_str_identifier+"*."+u_transport_var+".nc"
    vmo_vars_str=dir_base+dir_vars+"*"+file_str_identifier+"*."+v_transport_var+".nc"
    files_timestep=glob(f"{umo_vars_str}")
    files_timestep+=glob(f"{vmo_vars_str}")
    if calc_rho_flag==False:
        rho_vars_str=dir_base+dir_vars+"*"+file_str_identifier+"*."+rho_var+".nc"
        files_timestep+=glob(f"{rho_vars_str}")
    elif calc_rho_flag == True:
        theta_vars_str=dir_base+dir_vars+"*"+file_str_identifier+"*."+theta_var+".nc"
        salt_vars_str=dir_base+dir_vars+"*"+file_str_identifier+"*."+salt_var+".nc"
        files_timestep+=glob(f"{theta_vars_str}")
        files_timestep+=glob(f"{salt_vars_str}")
    
    ds = xr.open_mfdataset(files_timestep, decode_times=decode_times_flag)
    grid=xr.open_dataset(dir_base+dir_grid)

    ### If there are not the same number of dimensions of yq and yh, cut off the extra dimension:
    if len(ds[y_qpoint_1Dvar])==len(ds[y_hpoint_1Dvar])+1:
        ds = ds.isel( { y_qpoint_1Dvar: slice(1,len(ds[y_qpoint_1Dvar])) } )
        grid = grid.isel( { y_qpoint_1Dvar: slice(1,len(ds[y_qpoint_1Dvar])) } )
    elif len(ds[y_qpoint_1Dvar])==len(ds[y_hpoint_1Dvar]):
        'this is fine'
    else:
        raise Exception("yq must be more positive than yh or be 1 element larger than yh")
    if len(ds[x_qpoint_1Dvar])==len(ds[x_hpoint_1Dvar])+1:
        ds = ds.isel( { x_qpoint_1Dvar: slice(1,len(ds[x_qpoint_1Dvar])) } )
        grid = grid.isel( { x_qpoint_1Dvar: slice(1,len(ds[x_qpoint_1Dvar])) } )
    elif len(ds[x_qpoint_1Dvar])==len(ds[x_hpoint_1Dvar]):
        'this is fine'
    else:
        raise Exception("xq must be more positive than xh or be 1 element larger than xh")

    ### If the 1D coordinate names are not xh,xq, etc, then rename them to x_hpoint_1Dvar, x_qpoint_1Dvar etc
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
    
    ### Reduce xarray data domain to fit around chosen section coordinates
    lat_range_min=np.abs(ds.yh-(min(section_node_lats)-1)).argmin()
    lat_range_max=np.abs(ds.yh-(max(section_node_lats)+1)).argmin()
    lon_range_min=np.abs(ds.xh-(min(section_node_lons)-1)).argmin()
    lon_range_max=np.abs(ds.xh-(max(section_node_lons)+10)).argmin()
    ds_subpolar = ds.sel(yq=slice(ds.yq[lat_range_min],ds.yq[lat_range_max]),
                             xh=slice(ds.xh[lon_range_min],ds.xh[lon_range_max]),
                             yh=slice(ds.yh[lat_range_min],ds.yh[lat_range_max]),
                             xq=slice(ds.xq[lon_range_min],ds.xq[lon_range_max]))
    grid_subpolar = grid.sel(yq=slice(ds.yq[lat_range_min],ds.yq[lat_range_max]),
                             xh=slice(ds.xh[lon_range_min],ds.xh[lon_range_max]),
                             yh=slice(ds.yh[lat_range_min],ds.yh[lat_range_max]),
                             xq=slice(ds.xq[lon_range_min],ds.xq[lon_range_max]))
    if time_limits:
        ds_subpolar = ds_subpolar.sel(time=slice(time_limits[0],time_limits[1]))
    
    
    ### Run Raf's sectionate tool to extract T,S and V along chosen section coordinates
    isec, jsec, xsec, ysec = sectionate.create_section_composite(grid_subpolar['geolon_c'],
                                                                 grid_subpolar['geolat_c'],
                                                                 section_node_lons,
                                                                 section_node_lats)
    
    
    [corner_offset1,corner_offset2]=sectionate.find_offset_center_corner(grid_subpolar[lons_tpoint], grid_subpolar[lats_tpoint], 
                                                                         grid_subpolar[lons_cpoint], grid_subpolar[lats_cpoint])
    
    dsT = sectionate.MOM6_normal_transport(ds_subpolar, isec, jsec,utr=u_transport_var,vtr=v_transport_var,layer=z_layer_var,interface=z_inter_var,
                                           offset_center_x=corner_offset1,offset_center_y=corner_offset2,old_algo=True)
    transp=dsT.uvnormal
      
    if calc_rho_flag==False:
        thetao=[]
        so=[]
        rho = sectionate.MOM6_extract_hydro(ds_subpolar[rho_var], isec, jsec,
                                            offset_center_x=corner_offset1,offset_center_y=corner_offset2)-1000
    elif calc_rho_flag == True:
        thetao = sectionate.MOM6_extract_hydro(ds_subpolar[theta_var], isec, jsec,offset_center_x=corner_offset1,offset_center_y=corner_offset2)
        so = sectionate.MOM6_extract_hydro(ds_subpolar[salt_var], isec, jsec,
                                           offset_center_x=corner_offset1,offset_center_y=corner_offset2)
        rho = momlevel.derived.calc_rho(thetao,so,ref_pres)-1000
    
    rho = rho.rename('rho')
    
    if mom_gridwidth_algo:
        ds_grid_width = sectionate.sectionate_gridwidth(grid_subpolar,ds_subpolar,isec,jsec,interface=z_inter_var,layer=z_layer_var)
        cell_area = ds_grid_width.section_cellarea
        section_gridwidth = ds_grid_width.section_gridwidth
    else:
        earth_radius=6371000  # in km
        section_gridwidth=earth_radius*sectionate.distance_on_unit_sphere(ysec[0:-1],xsec[0:-1],ysec[1:],xsec[1:])
        depth_diff=np.diff(dsT[z_inter_var])
        cell_area=np.repeat(np.expand_dims(section_gridwidth, axis = 0),np.shape(dsT[z_layer_var])[0],axis=0)*np.repeat(np.expand_dims(depth_diff, axis = 1),np.shape(section_gridwidth)[0],axis=1)
        # cell_area=np.repeat(np.expand_dims(cell_area,axis=0),np.shape(dsT.time)[0],axis=0)
        cell_area=xr.DataArray(data=cell_area, coords={z_layer_var : transp[z_layer_var], 'sect' : transp['sect']}, dims=(z_layer_var,'sect')).drop_vars('sect')

    
    return dsT, transp, rho, xsec, ysec, cell_area, so, thetao, section_gridwidth, grid_subpolar


def SigmaZ_diag_ComputeDiag(transp,rho,rholims,depth,rebin_depth=[],nrho=500,z_layer_var='z_l',dist_var='sect',numpy_algo=False):

    rho0_ref=rholims[0]+np.arange(0,nrho-1)*((rholims[1]-rholims[0])/nrho); 
    rho0_bounds=rholims[0]-(rho0_ref[1]-rho0_ref[0])/2+np.arange(0,nrho)*((rholims[1]-rholims[0])/nrho); #the grid edges of the density vertical axis

    ### create the transport SigmaZ matrix along the section
    if numpy_algo:
        
        model_depth=depth[1:]  
        depth_diff=np.diff(depth)
        transp[np.isnan(transp)]=0
        rho[np.isnan(rho)]=0
        ty_z_rho=make_sigma_z(rho0_bounds,transp,rho)
        ty_z_rho=ty_z_rho
        ty_z_rho[np.isnan(ty_z_rho)] = 0

    else:
        model_depth=depth.isel({ z_layer_var : slice(1,len(depth))})
        depth_diff=depth.diff(z_layer_var)
        ty_z_rho=histogram(rho,bins=[rho0_bounds],weights=transp.fillna(0.),dim=[dist_var])
    
    
    if len(rebin_depth)>0:
        if numpy_algo:
            ty_z_rho_rebin=rebin_sigma_z(model_depth,depth_diff,rebin_depth,ty_z_rho)
        else:
            ty_z_rho_rebin=rebin_sigma_z(model_depth.values,depth_diff.values,rebin_depth,ty_z_rho.values)
            ty_z_rho_rebin=xr.DataArray(data=ty_z_rho_rebin, dims=('time','z_l_rebin','rho_bin'), 
                                        coords={'time' : ty_z_rho['time'],'z_l_rebin' : rebin_depth, 'rho_bin' : ty_z_rho['rho_bin']})
    else:
        ty_z_rho_rebin=[]
    
    return ty_z_rho, ty_z_rho_rebin, rho0_bounds, rho0_ref


def SigmaZ_diag_PlotDiag(ty_z_rho,depth,rho0_ref,rho0_bounds,rho0=1030,plot_flag=True,sig_axis_plot_lims=[26.5,28],z_axis_plot_lims=[100,4000],z_dim='z_l_rebin',rho_dim='rho_bin',numpy_algo=False):
    
    if numpy_algo:
        ty_z = ty_z_rho.sum(axis=2)
        ty_rho = ty_z_rho.sum(axis=1)
        MOCz = np.cumsum(ty_z,axis=1)
        MOCrho = np.cumsum(ty_rho,axis=1)
        MOCrho_ts = MOCrho.argmax(axis=1)
        MOCz_ts = MOCz.argmax(axis=1)
    else:
        ty_z = ty_z_rho.sum(dim=rho_dim)
        ty_rho = ty_z_rho.sum(dim=z_dim)
        MOCz = ty_z.cumsum(dim=z_dim)
        MOCrho = ty_rho.cumsum(dim=rho_dim)
        MOCrho_ts = MOCrho.max(dim=rho_dim)
        MOCz_ts = MOCz.max(dim=z_dim)


    if plot_flag:
        fig = plt.figure(figsize=(12,7))
        ax = fig.add_subplot(2,2,3)
        ax.plot(MOCz.mean(axis=0)/rho0/1e6,-depth)
        ax.set_title('Xsection MOCz profile')
        ax.set_ylabel('z (m)')
        ax.set_xlabel('V (Sv)')
        plt.ylim([-z_axis_plot_lims[1],-z_axis_plot_lims[0]])
        ax = fig.add_subplot(2,2,1)
        ax.plot(MOCrho.mean(axis=0)/rho0/1e6,-rho0_ref)
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
        ch1=ax0.pcolormesh(rho0_bounds[0:-1],-depth,ty_z_rho.mean(axis=0)/rho0/1e6,shading='flat',vmin=-5e-1,vmax=5e-1,cmap='RdBu_r')
        ax0.set_title('Xsec transp. in Sigma-Z')
        ax0.set_xlabel('Sigma (kg/m3)')
        plt.xlim( [sig_axis_plot_lims[0],sig_axis_plot_lims[1]] ), plt.ylim( [-z_axis_plot_lims[1],-z_axis_plot_lims[0]] )
        box = ax0.get_position()
        ax0.set_position([box.x0*1, box.y0, box.width, box.height])
        axColor = plt.axes([box.x0*1 + box.width * 1.05, box.y0, 0.01, box.height])
        plt.colorbar(ch1, cax = axColor, orientation="vertical")
        #plt.show()
        ax1 = fig.add_subplot(spec[1])
        ax1.plot(rho0_ref,ty_rho.mean(axis=0)/rho0/1e6)
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

def SigmaZ_diag_computeMFTMHT(ty_z_rho,transp,so,thetao,rho,cell_area,rho0_bounds,rho_dim='rho_bin',z_dim='z_l',dist_var='sect',annual_mean_flag=False,plot_flag=True):
    ### Calculate density-space MFT and MHT components

    if type(ty_z_rho)==np.ndarray:
        ty_z_rho=xr.DataArray(data=ty_z_rho,dims=('time','z_l','rho_bin')).rename('histogram_rho')
        transp=xr.DataArray(data=transp,dims=('time','z_l','sect')).rename('uvnormal')
        so=xr.DataArray(data=so,dims=('time','z_l','sect')).rename('so')
        thetao=xr.DataArray(data=thetao,dims=('time','z_l','sect')).rename('thetao')
        rho=xr.DataArray(data=rho,dims=('time','z_l','sect')).rename('rho')
        cell_area=xr.DataArray(data=cell_area,dims=('time','z_l','sect')).rename('section_cellarea')
        
    Cp=3850 # heat capacity of seawater      

    ### calculate all tracer SigmaZ matrices 
    thetao_z_rho = histogram(rho,bins=[rho0_bounds],weights=(thetao*cell_area).fillna(0.),dim=[dist_var])
    so_z_rho = histogram(rho,bins=[rho0_bounds],weights=(so*cell_area).fillna(0.),dim=[dist_var])
    cellarea_z_rho = histogram(rho,bins=[rho0_bounds],weights=(cell_area).fillna(0.),dim=[dist_var])
    thetao_z_rho_mean = thetao_z_rho/cellarea_z_rho  # The mean temperature in each sigma_z cell for OSNAP 
    so_z_rho_mean = so_z_rho/cellarea_z_rho   # The mean salinity in each sigma_z cell for OSNAP 

    #### Calculate z- and rho- space zonal means of T,S and V, for use in calculating overturning component of MFT and MHT. 
    S_bar=so_z_rho.fillna(0).sum()/cellarea_z_rho.fillna(0).sum() 
    S_zm_z = so_z_rho.fillna(0).sum(dim=rho_dim)/cellarea_z_rho.fillna(0).sum(dim=rho_dim) 
    S_zm_rho = so_z_rho.fillna(0).sum(dim=z_dim)/cellarea_z_rho.fillna(0).sum(dim=z_dim) # rho-space zonal mean salinity
    theta_zm_z = thetao_z_rho.fillna(0).sum(dim=rho_dim)/cellarea_z_rho.fillna(0).sum(dim=rho_dim) 
    theta_zm_rho = thetao_z_rho.fillna(0).sum(dim=z_dim)/cellarea_z_rho.fillna(0).sum(dim=z_dim) 
    ty_zm_z = ty_z_rho.fillna(0).sum(dim=rho_dim)
    ty_zm_rho = ty_z_rho.fillna(0).sum(dim=z_dim)

    ### calculate MOC MFT
    MFT_MOCrho =- ty_zm_rho*(S_zm_rho-S_bar)/S_bar
    MFT_MOCz =- ty_zm_z*(S_zm_z-S_bar)/S_bar
    MFT_MOCrho_sum = MFT_MOCrho.fillna(0).sum(dim=rho_dim)
    MFT_MOCz_sum = MFT_MOCz.fillna(0).sum(dim=z_dim)
    ### calculate MOC MHT
    MHT_MOCrho = Cp*ty_zm_rho*theta_zm_rho
    MHT_MOCz = Cp*ty_zm_z*theta_zm_z
    MHT_MOCrho_sum = MHT_MOCrho.fillna(0).sum(dim=rho_dim)
    MHT_MOCz_sum = MHT_MOCz.fillna(0).sum(dim=z_dim)

    ### calculate the transport*tracer SigmaZ matrices, and calculate (zonal- and total-) integrated MFT 
    Vthetao_z_rho = histogram(rho,bins=[rho0_bounds],weights=(thetao*transp).fillna(0.),dim=[dist_var])
    Vso_z_rho = histogram(rho,bins=[rho0_bounds],weights=((so-S_bar)*transp/S_bar).fillna(0.),dim=[dist_var])
    
    MFT_zonmean_rho=-Vso_z_rho.fillna(0).sum(dim=z_dim)
    MFT_zonmean_z=-Vso_z_rho.fillna(0).sum(dim=rho_dim)
    MFT_GYRErho=MFT_zonmean_rho-MFT_MOCrho
    MFT_GYREz=MFT_zonmean_z-MFT_MOCz
    MFT_GYRErho_sum=MFT_GYRErho.fillna(0).sum(dim=rho_dim)
    MFT_GYREz_sum=MFT_GYREz.fillna(0).sum(dim=z_dim)
    MFT_sum=-Vso_z_rho.fillna(0.).sum(dim=[z_dim,rho_dim])
    MHT_zonmean_rho=Cp*Vthetao_z_rho.fillna(0).sum(dim=z_dim)
    MHT_zonmean_z=Cp*Vthetao_z_rho.fillna(0).sum(dim=rho_dim)
    MHT_GYRErho=MHT_zonmean_rho-MHT_MOCrho
    MHT_GYREz=MHT_zonmean_z-MHT_MOCz
    MHT_GYRErho_sum=MHT_GYRErho.fillna(0).sum(dim=rho_dim)
    MHT_GYREz_sum=MHT_GYREz.fillna(0).sum(dim=z_dim)
    MHT_sum=Cp*Vthetao_z_rho.fillna(0.).sum(dim=[z_dim,rho_dim])

    MOCrho =ty_z_rho.sum(dim=z_dim).cumsum(dim=rho_dim).max(dim=rho_dim)
    MOCz = ty_z_rho.sum(dim=rho_dim).cumsum(dim=z_dim).max(dim=z_dim)
    
    ### Combine output into single datasets for sigma-, z-, and SigmaZ space
    MOCsig_MHTMFT = xr.Dataset()
    MOCsig_MHTMFT['MOCrho']=MOCrho
    MOCsig_MHTMFT['MHT_zonmean_rho']=MHT_zonmean_rho
    MOCsig_MHTMFT['MHT_sum']=MHT_sum
    MOCsig_MHTMFT['MHT_MOCrho']=MHT_MOCrho
    MOCsig_MHTMFT['MHT_MOCrho_sum']=MHT_MOCrho_sum
    MOCsig_MHTMFT['MHT_GYRErho']=MHT_GYRErho
    MOCsig_MHTMFT['MHT_GYRErho_sum']=MHT_GYRErho_sum
    MOCsig_MHTMFT['MFT_sum']=MFT_sum
    MOCsig_MHTMFT['MFT_MOCrho']=MFT_MOCrho
    MOCsig_MHTMFT['MFT_MOCrho_sum']=MFT_MOCrho_sum
    MOCsig_MHTMFT['MFT_GYRErho']=MFT_GYRErho
    MOCsig_MHTMFT['MFT_GYRErho_sum']=MFT_GYRErho_sum

    MOCz_MHTMFT = xr.Dataset()
    MOCz_MHTMFT['MOCz']=MOCz
    MOCz_MHTMFT['MHT_zonmean_z']=MHT_zonmean_z
    MOCz_MHTMFT['MHT_sum']=MHT_sum
    MOCz_MHTMFT['MHT_MOCz']=MHT_MOCz
    MOCz_MHTMFT['MHT_MOCz_sum']=MHT_MOCz_sum
    MOCz_MHTMFT['MHT_GYREz']=MHT_GYREz
    MOCz_MHTMFT['MHT_GYREz_sum']=MHT_GYREz_sum
    MOCz_MHTMFT['MFT_sum']=MFT_sum
    MOCz_MHTMFT['MFT_MOCz']=MFT_MOCz
    MOCz_MHTMFT['MFT_MOCz_sum']=MFT_MOCz_sum
    MOCz_MHTMFT['MFT_GYREz']=MFT_GYREz
    MOCz_MHTMFT['MFT_GYREz_sum']=MFT_GYREz_sum

    MOCsigz_MHTMFT = xr.Dataset()
    MOCsigz_MHTMFT['Vthetao_z_rho']=Vthetao_z_rho
    MOCsigz_MHTMFT['Vso_z_rho']=Vso_z_rho 
    
    ## temporally rebin the monthly offline timeseries into annual ones
    if annual_mean_flag:
        MOCsig_MHTMFT=MOCsig_MHTMFT.coarsen(time=12).mean()
        MOCz_MHTMFT=MOCz_MHTMFT.coarsen(time=12).mean()
        MOCsigz_MHTMFT=MOCsigz_MHTMFT.coarsen(time=12).mean()
        
    if plot_flag:
        rho0=1030
        fig = plt.figure(figsize=(10,12))
        ax = fig.add_subplot(3,1,1)
        ax.plot(MOCsig_MHTMFT.MOCrho.time,MOCsig_MHTMFT.MOCrho/rho0/1e6)
        ax.set_title('MOC in rho-space')
        ax.set_ylabel('Sv')
        ax = fig.add_subplot(3,1,2)
        ax.plot(MOCsig_MHTMFT.MOCrho.time,(MOCsig_MHTMFT.MHT_sum)/1e15)
        ax.plot(MOCsig_MHTMFT.MOCrho.time,(MOCsig_MHTMFT.MHT_MOCrho_sum)/1e15)
        ax.plot(MOCsig_MHTMFT.MOCrho.time,(MOCsig_MHTMFT.MHT_GYRErho_sum)/1e15)
        ax.set_title('MHT in rho-space')
        ax.set_ylabel('PW')
        ax = fig.add_subplot(3,1,3)
        ax.plot(MOCsig_MHTMFT.MOCrho.time,(MOCsig_MHTMFT.MFT_sum)/rho0/1e6,label='TOTrho')
        ax.plot(MOCsig_MHTMFT.MOCrho.time,(MOCsig_MHTMFT.MFT_MOCrho_sum)/rho0/1e6,label='MOCrho')
        ax.plot(MOCsig_MHTMFT.MOCrho.time,(MOCsig_MHTMFT.MFT_GYRErho_sum)/rho0/1e6,label='GYRErho')
        ax.legend(loc="lower left")
        ax.set_title('MFT in rho-space')
        ax.set_ylabel('Sv')
        ax.set_xlabel('time (months)')

            
    return MOCsig_MHTMFT, MOCz_MHTMFT, MOCsigz_MHTMFT


@njit
def make_sigma_z(rho0_bounds,transp_vals,rho_vals):
    ty_z_rho=np.zeros((np.shape(transp_vals)[0],np.shape(transp_vals)[1],np.shape(rho0_bounds)[0]-1));
    for k in range(0,np.shape(transp_vals)[0]):
        for krho in range(0,np.shape(rho0_bounds)[0]-1):
           for kk in range(0,np.shape(transp_vals)[1]): 
               for ii in range(0,np.shape(transp_vals)[2]): 
                   if (rho_vals[k,kk,ii] >= rho0_bounds[krho]) & (rho_vals[k,kk,ii]<rho0_bounds[krho+1]):
                       ty_z_rho[k,kk,krho]=ty_z_rho[k,kk,krho]+transp_vals[k,kk,ii]
    return ty_z_rho


@njit
def rebin_sigma_z(model_depth,depth_diff,rebin_depth,ty_z_rho):
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

