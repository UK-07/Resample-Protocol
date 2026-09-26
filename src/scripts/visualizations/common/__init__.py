"""Shared code of the paper's plot scripts (``src/scripts/visualizations/<plot>/{gather,plot}.py``).

``paths``           where the released paper tree and the plot outputs live (``--cueball-dir`` resolution)
``ssp_common``      the single-sample-protocol stitching (manifest + binary judge + baseline sample 0 +
                    re-roll reliance), its row cache and the ``gather_cli`` every survival gather runs
``section1_plot``   grouped Wilson bars and the plot-side CLI of the survival figures
``alpha_common``    re-aggregation of the alpha-vs-measured-noise rows, chi-square tests, self-test
``section3_claims`` the judge-role pool of the claims-vs-behaviour figures
``rates_common``    single-sample vs resampled rates (dataset_B), cell tables, ``gather_common``
``yield_common``    filtered manifest loaders, the RSP-unfaithful pool, the cluster bootstrap
"""
