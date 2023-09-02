import os
import time
import requests
import pandas as pd
from astropy.time import Time
from datetime import datetime

def get_time_radius(url):
    r = requests.get(url).text
    find_date = r.rfind("NOTICE_DATE:", 0, r.find("LOC_URL:"))
    find_radius = r.rfind("GRB_ERROR:", 0, r.find("LOC_URL:"))
    date = r[find_date+17:find_date+17+22]
    radius = r[find_radius+17:find_radius+17+5]
    return date, radius

def tiling():
    basepath = "https://gcn.gsfc.nasa.gov/other/"
    triggers = pd.DataFrame(pd.read_csv("/home/ravioli/astro/git/fermi_grbs/u_triggers_with_probs.csv"))
    for trigger, name, healpix in zip(triggers['Trig_no'], triggers['GRBName'], triggers['healpix']):
        in_file = f"/home/ravioli/astro/git/fermi_grbs/data/glg_healpix_all_{name}.fit"
        if healpix != 0:
            with open("/home/ravioli/astro/git/fermi_grbs/coverage_4h.csv", "a") as f:
                f.write(f"{None},{trigger},{None},{None},{None}\n")
            print("NO HEALPIX", name, trigger)
        else:
            url = f"{basepath}/{trigger}.fermi"
            dateradius = pd.DataFrame(pd.read_csv("/home/ravioli/astro/git/fermi_grbs/coverage_old.csv"))
            date, radius = dateradius.loc[dateradius['trig_no'] == trigger, ['time', 'radius']].values[0]
            # date, radius = get_time_radius(url)
            # time = Time(datetime.strptime(date, "%a %d %b %y %H:%M:%S"), format="datetime", scale="utc")
            tile_cmd = f"python3 program_for_emgw_mapping_4h.py \
                -input_file {in_file} \
                -time '{date}' -trig_no {trigger} \
                -radius {radius}"
            r = int(os.system(tile_cmd))
            if r != 0:
                print("GRB BELOW -32.5 DECLINATION")
                with open("/home/ravioli/astro/git/fermi_grbs/coverage_4h.csv", "a") as f:
                    f.write(f"{os.path.basename(in_file)},{trigger},{date},{radius},NOT VISIBLE\n")
            print(name, trigger, date, radius, r)

if __name__ == "__main__":
    t = time.time()
    tiling()
    print("######################################\n######################################\n")
    print("Total time taken: ", time.time()-t)
    print("######################################\n######################################\n")