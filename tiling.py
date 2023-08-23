import os
import requests
import pandas as pd

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
            with open("/home/ravioli/astro/git/fermi_grbs/coverage.csv", "a") as f:
                f.write(f"{None},{trigger},{None},{None}\n")
            f.close()
            print("NO HEALPIX", name, trigger)
        else:
            url = f"{basepath}/{trigger}.fermi"
            date, radius = get_time_radius(url)
            tile_cmd = f"python3 program_for_emgw_mapping.py \
                -input_file /home/ravioli/astro/git/fermi_grbs/data/glg_healpix_all_{name}.fit \
                -time '{date}' -trig_no {trigger}"
            r = os.system(tile_cmd)
            print(name, trigger, date, radius, r)

if __name__ == "__main__":
    tiling()