""" Pipelines expressed as dask bags
"""

from dask import bag

from arl.data.parameters import get_parameter
from arl.graphs.bags import deconvolve_bag, invert_bag, predict_bag, residual_image_bag, \
    restore_bag, calibrate_vis_bag, qa_visibility_bag
from arl.visibility.operations import subtract_visibility, copy_visibility
from arl.visibility.coalesce import coalesce_visibility, decoalesce_visibility, \
    convert_visibility_to_blockvisibility, convert_blockvisibility_to_visibility


def continuum_imaging_pipeline_bag(vis_bag, model_bag, context, **kwargs) -> bag:
    """ Create bag for the continuum imaging pipeline.
    
    Same as ICAL but with no selfcal.
    
    :param vis_bag:
    :param model_bag:
    :param context: Imaging context
    :param kwargs: Parameters for functions in bags
    :return:
    """
    psf_bag = invert_bag(vis_bag, model_bag, dopsf=True, context=context, **kwargs)
    res_bag = residual_image_bag(vis_bag, model_bag, context=context, **kwargs)
    deconvolve_model_bag = deconvolve_bag(res_bag, psf_bag, model_bag, **kwargs)
    
    nmajor = get_parameter(kwargs, "nmajor", 5)
    if nmajor > 1:
        for cycle in range(nmajor):
            res_bag = residual_image_bag(vis_bag, deconvolve_model_bag, context=context,
                                         **kwargs)
            deconvolve_model_bag = deconvolve_bag(res_bag, psf_bag, deconvolve_model_bag,
                                                  **kwargs)
    
    res_bag = residual_image_bag(vis_bag, deconvolve_model_bag, context=context, **kwargs)
    rest_bag = restore_bag(deconvolve_model_bag, psf_bag, res_bag, **kwargs)
    return rest_bag


def spectral_line_imaging_pipeline_bag(vis_bag, model_bag,
                                       continuum_model_bag=None,
                                       context='2d',
                                       **kwargs) -> bag:
    """Create bag for spectral line imaging pipeline

    Uses the ical pipeline after subtraction of a continuum model
    
    :param vis_bag: List of visibility bags
    :param model_bag: Spectral line model bag
    :param continuum_model_bag: Continuum model bag
    :param kwargs: Parameters for functions in bags
    :return: bags of (deconvolved model, residual, restored)
    """
    if continuum_model_bag is not None:
        vis_bag = predict_bag(vis_bag, continuum_model_bag, **kwargs)
    
    return ical_pipeline_bag(vis_bag, model_bag, context=context, first_selfcal=None,
                             **kwargs)


def ical_pipeline_bag(vis_bag, model_bag, context='2d', first_selfcal=None,
                      **kwargs) -> bag:
    """Create bag for ICAL pipeline
    
    :param vis_bag:
    :param model_bag:
    :param context: Imaging context
    :param kwargs: Parameters for functions in bags
    :return:
    """
    block_vis_bag = bag.from_sequence(vis_bag.compute())\
        .map(convert_visibility_to_blockvisibility)
    
    psf_bag = invert_bag(vis_bag, model_bag, context=context, dopsf=True, **kwargs)
    
    model_vis_bag = bag.from_sequence(vis_bag.map(copy_visibility, zero=True).compute())
    model_vis_bag = predict_bag(model_vis_bag, model_bag, context=context, **kwargs)

    if first_selfcal is not None and first_selfcal == 0:
        block_model_vis_bag = model_vis_bag.map(convert_visibility_to_blockvisibility)
        block_vis_bag = calibrate_vis_bag(block_vis_bag, block_model_vis_bag, **kwargs)
    
    vis_bag = block_vis_bag.map(convert_blockvisibility_to_visibility)

    res_vis_bag = vis_bag.map(subtract_visibility, model_vis_bag)
    
    res_bag = invert_bag(res_vis_bag, model_bag, context=context, dopsf=False, **kwargs)
    deconvolve_model_bag = deconvolve_bag(res_bag, psf_bag, model_bag, **kwargs)
    
    nmajor = get_parameter(kwargs, "nmajor", 5)
    if nmajor > 1:
        for cycle in range(nmajor):
            model_vis_bag = model_vis_bag.map(copy_visibility, zero=True)
            model_vis_bag = predict_bag(model_vis_bag, deconvolve_model_bag, context=context, **kwargs)
            
            if first_selfcal is not None and cycle >= first_selfcal:
                block_model_vis_bag = model_vis_bag.map(convert_visibility_to_blockvisibility)
                block_vis_bag = calibrate_vis_bag(block_vis_bag, block_model_vis_bag, **kwargs)
            
            vis_bag = block_vis_bag.map(convert_blockvisibility_to_visibility)

            res_vis_bag = vis_bag.map(subtract_visibility, model_vis_bag)
            res_bag = invert_bag(res_vis_bag, model_bag, context=context, dopsf=False, **kwargs)
            deconvolve_model_bag = deconvolve_bag(res_bag, psf_bag, deconvolve_model_bag,
                                                  **kwargs)
    res_bag = residual_image_bag(vis_bag, deconvolve_model_bag, context=context, **kwargs)
    rest_bag = restore_bag(deconvolve_model_bag, psf_bag, res_bag, **kwargs)
    return rest_bag
