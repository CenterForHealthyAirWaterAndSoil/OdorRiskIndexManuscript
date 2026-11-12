"""Fill the Open-Meteo BLH gap (Jan-Jun 2024) at the Calvert monitors from Copernicus ERA5 (same reanalysis)."""
import cdsapi, os
out = "era5_blh_calvert_2024h1.nc"
if not os.path.exists(out):
    cdsapi.Client().retrieve("reanalysis-era5-single-levels", {
        "product_type": ["reanalysis"], "variable": ["boundary_layer_height"], "year": ["2024"],
        "month": ["01", "02", "03", "04", "05", "06", "07"], "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)], "data_format": "netcdf", "download_format": "unarchived",
        "area": [37.3, -88.6, 36.8, -88.1]}, out)
print("done", os.path.getsize(out))
