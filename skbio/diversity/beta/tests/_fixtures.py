# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------

"""Shared QIIME 1.9.1 tiny-test fixture loading for unifrac tests."""

import pandas as pd

from skbio import TreeNode, DistanceMatrix
from skbio.util._testing import get_data_path


class QiimeTinyTestMixin:
    """Mixin providing access to the qiime-191-tt fixture directory."""

    def _load_qiime_191_tt(self):
        base = get_data_path('qiime-191-tt', subfolder='data')
        df = pd.read_csv(f'{base}/otu-table.tsv', sep='\t', skiprows=1, index_col=0)
        taxa = df.index.astype(str).tolist()
        sample_ids = df.columns.tolist()
        table = df.T.values
        tree = TreeNode.read(f'{base}/tree.nwk')
        return table, taxa, tree, sample_ids

    def _load_dm_fixture(self, filename):
        base = get_data_path('qiime-191-tt', subfolder='data')
        return DistanceMatrix.read(f'{base}/{filename}')
