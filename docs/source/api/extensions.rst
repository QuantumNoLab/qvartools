GPU Extensions (Experimental)
=============================

.. module:: qvartools._ext

The ``_ext`` subpackage provides experimental GPU-accelerated backends.
These modules are optional and do not affect core functionality if their
external dependencies are unavailable.

sbd Subprocess
--------------

GPU-native Selected Basis Diagonalisation via the external ``sbd`` binary
(r-ccs-cms/sbd). Requires a compiled ``sbd`` binary and MPI runtime.

.. autofunction:: qvartools._ext.sbd_subprocess.sbd_available

.. autofunction:: qvartools._ext.sbd_subprocess.sbd_diagonalize

CUDA-QX VQE
------------

VQE and ADAPT-VQE pipeline wrapper using CUDA-QX Solvers.
Requires CUDA-Q >= 0.14 and CUDA-QX Solvers >= 0.5.

.. autofunction:: qvartools._ext.cudaq_vqe.run_cudaq_vqe
