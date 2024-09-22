""" Parameter dict for BING 2.0 """

from collections import namedtuple


def p_ntuple(model_names:list,
               wv_min:float=400., 
               wv_max:float=700., 
               satellite:str='PACE',
               scl_noise:float=None,
               add_noise:bool=False,
               set_Sdg:bool=None,
               sSdg:float=None,
               beta:float=None, 
               nMC:int=None,
    ):

    pdict = dict(model_names=model_names,
                    wv_min=wv_min,
                    wv_max=wv_max,
                    satellite=satellite,
                    add_noise=add_noise,
                    set_Sdg=set_Sdg,
                    sSdg=sSdg,
                    nMC=nMC,
                    beta=beta)
    # Scale noise
    if scl_noise is not None:
        pdict['scl_noise'] = scl_noise
    else:
        pdict['scl_noise'] = satellite
    #
    MyNamedTuple = namedtuple('BING20_tuple', pdict.keys())
    p = MyNamedTuple(**pdict)
    #
    return p

    