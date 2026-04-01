
import numpy as np
from correct_atmosphere import downwelling
from bing.fitting import l23

wave = np.linspace(400, 700., 300)
Ed = downwelling.downwelling_irradiance(wave, 0.)
#Ed = downwelling.downwelling_irradiance(a_model.wave, 0.)
#embed(header='934 of test')


