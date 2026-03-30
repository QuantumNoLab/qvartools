End-to-End Method Pipelines
===========================

.. module:: qvartools.methods

The ``methods`` subpackage provides complete pipelines combining NQS training,
configuration sampling, and iterative diagonalisation with eigenvector feedback.

NQS + SQD
---------

.. autoclass:: qvartools.methods.nqs.nqs_sqd.NQSSQDConfig
   :members:

.. autofunction:: qvartools.methods.nqs.nqs_sqd.run_nqs_sqd

NQS + SKQD
----------

.. autoclass:: qvartools.methods.nqs.nqs_skqd.NQSSKQDConfig
   :members:

.. autofunction:: qvartools.methods.nqs.nqs_skqd.run_nqs_skqd

HI + NQS + SQD (Iterative)
---------------------------

.. autoclass:: qvartools.methods.nqs.hi_nqs_sqd.HINQSSQDConfig
   :members:

.. autofunction:: qvartools.methods.nqs.hi_nqs_sqd.run_hi_nqs_sqd

HI + NQS + SKQD (Iterative)
----------------------------

.. autoclass:: qvartools.methods.nqs.hi_nqs_skqd.HINQSSKQDConfig
   :members:

.. autofunction:: qvartools.methods.nqs.hi_nqs_skqd.run_hi_nqs_skqd
