# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import TestCase, main

from skbio.diversity.beta._unifrac_gpu import detect_gpu_backend, get_cuda_module


class GpuBackendTests(TestCase):

    def test_detect_gpu_backend_returns_known_value(self):
        self.assertIn(detect_gpu_backend(), ("cuda", "hip", None))

    def test_get_cuda_module_raises_cleanly_when_no_gpu(self):
        backend = detect_gpu_backend()
        if backend is not None:
            self.skipTest("a GPU backend is available in this environment")
        with self.assertRaises(ImportError):
            get_cuda_module()


if __name__ == "__main__":
    main()
