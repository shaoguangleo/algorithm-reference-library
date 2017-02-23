.. Fourier processing

Fourier processing
******************

There are many algorithms for imaging, using different approaches to correct for various effects:

+ Simple 2D transforms
+ Partitioned image (i.e. faceted) and uv transforms
+ W projection
+ W snapshots
+ W slices
+ A projection variants
+ Visibility coalescence and de-coalescence
+ MFS variants
+ Differential residuals calculations

Since the scale of SDP is so much larger than previous telescopes, it is not clear which scaling strategies and
algorithms are going to offer the best performance. For this reason, it is important the synthesis framework not be
restrictive.

All the above functions are linear in the visibilities and image. The 2D transform is correct for sufficiently
restricted context. Hence we will layer all algorithms on top of the 2D transform. This means that a suitable
framework decomposes the overall transform into suitable linear combinations of invocations of 2D transforms. We can
use python iterators to perform the subsectioning. For example, the principal image iteration via a raster
implemented by a python generator::

        m31model=create_test_image()
        for ipatch in raster(m31model, nraster=2):
            # each image patch can be used to add to the visibility data
            vis + = predict_2d(vis, ipatch, params)

        # For image partitioning and snapshot processing
        iraster, interval = find_optimum_iraster_times(vis, model)
        m31model=create_test_image()
        for ipatch in raster(m31model, nraster=iraster):
            for subvis in snapshot(vis, interval=interval):
                # each patch can be used to add to the visibility data
                subvis + = predict_2d(subvis, ipatch, params)

This relies upon the data objects (model and vis) possessing sufficient meta data to enable operations such as phase
rotation from one frame to another.

In addition, iteration through the visibility data must tbe varied:

+ By time
+ By frequency
+ By w
+ By parallactic angle

The Visibility API should support these forms of iteration.

The pattern used in these algorithms is abstracted in the following diagram:

.. image:: ./ARL_fourier_processing.png
      :width: 1024px

The layering of predict and invert classes is shown below:

.. image:: ARL_predict_layering.png
      :width: 1024px

.. image:: ARL_invert_layering.png
      :width: 1024px

The top level functions are in green. All capability is therefore layered on two functions, predict_2d and invert_2d.


Parallel processing
*******************

ARL uses parallel processing to speed up some calculations. It is not intended to indicate a preference for how
parallel processing should be implemented in SDP.

We use an openMP-like package `pypm <https://github.com/classner/pymp/>`_. An example is to be found in
arl/fourier_transforms/invert_with_vis_iterator. The data are divided into timeslices and then processed in parallel::

      def invert_with_vis_iterator(vis, im, dopsf=False, vis_iter=vis_slice_iter, invert=invert_2d, **kwargs):
          """ Invert using a specified iterator and invert

          This knows about the structure of invert in different execution frameworks but not
          anything about the actual processing. This version support pymp and serial processing

          :param vis:
          :param im:
          :param dopsf:
          :param kwargs:
          :return:
          """
          resultimage = create_empty_image_like(im)

          nproc = get_parameter(kwargs, "nprocessor", 1)
          if nproc == "auto":
              nproc = multiprocessing.cpu_count()
          inchan, inpol, _, _ = im.data.shape
          totalwt = numpy.zeros([inchan, inpol], dtype='float')
          if nproc > 1:
              # We need to tell pymp that some arrays are shared
              resultimage.data = pymp.shared.array(resultimage.data.shape)
              resultimage.data *= 0.0
              totalwt = pymp.shared.array([inchan, inpol])

              # Extract the slices and run  on each one in parallel
              nslices = 0
              rowses = []
              for rows in vis_iter(vis, **kwargs):
                  nslices += 1
                  rowses.append(rows)

              log.debug("invert_iteratoe: Processing %d chunks %d-way parallel" % (nslices, nproc))
              with pymp.Parallel(nproc) as p:
                  for index in p.range(0, nslices):
                      visslice = create_visibility_from_rows(vis, rowses[index])
                      workimage, sumwt = invert(visslice, im, dopsf, **kwargs)
                      resultimage.data += workimage.data
                      totalwt += sumwt

          else:
              # Do each slice in turn
              i = 0
              for rows in vis_iter(vis, **kwargs):
                  visslice = create_visibility_from_rows(vis, rows)
                  workimage, sumwt = invert(visslice, im, dopsf, **kwargs)
                  resultimage.data += workimage.data
                  totalwt += sumwt
                  i += 1
          return resultimage, totalwt

