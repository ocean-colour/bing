""" Items related to MODIS """


# MODIS Aqua -- derived from https://seabass.gsfc.nasa.gov/search/?search_type=Perform%20Validation%20Search&val_sata=1&val_products=11&val_source=0
#  See MODIS_error.ipynb
#wv: 412, std=0.00141 sr^-1, rel_std=10.88%
#wv: 443, std=0.00113 sr^-1, rel_std=9.36%
#wv: 488, std=0.00113 sr^-1, rel_std=0.68%
#wv: 531, std=0.00102 sr^-1, rel_std=0.19%
#wv: 547, std=0.00117 sr^-1, rel_std=0.19%
#wv: 555, std=0.00120 sr^-1, rel_std=0.22%
#wv: 667, std=0.00056 sr^-1, rel_std=6.22%
#wv: 678, std=0.00060 sr^-1, rel_std=4.15%

modis_wave = [412, 443, 469, 488, 531, 547, 555, 645, 667, 678, 748]# , 859, 869] # nm
modis_aqua_error = [0.00141, 0.00113, 
                    0.00113,  # Assumed for 469
                    0.00113, 0.00102, 0.00117, 0.00120, 
                    0.00070,  # Assumed for 645
                    0.00056, 0.00060,
                    0.00060,  # Assumed for 748
                    ]