========================================
pybrium: Strategy and equilibria toolkit
========================================

pybrium is a Python library for calculations invoing strategies and equilibria.

Installation
============

Install with the following one-liner::

    $ pip install git+https://github.com/jma127/pybrium.git

Quickstart
==========

Calculate the maximum-entropy equilibrium and Nash averaged ratings for a three-way tournament::

    import torch
    from pybrium import nash_average

    winrates = torch.tensor([
        [0.5, 0.3, 0.5],
        [0.7, 0.5, 0.5],
        [0.5, 0.5, 0.5],
    ], dtype=torch.float32)
    logits = torch.log(winrates) - torch.log(1.0 - winrates)
    nash_equilibrium, nash_rating = nash_average(logits, steps=(2 ** 14))

See ``examples.py`` for more usage examples.

Contributing
============

Bugfixes and contributions are very much appreciated! Feel free to submit a pull request at any time.

Citing
======

If you use this library in your research, please consider citing it as follows::

    @misc{ma2019pybrium,
      author = {Jerry Ma},
      title = {pybrium: Strategy and equilibria toolkit},
      year = {2019},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/jma127/pybrium}}
    }

License
=======

This source code is licensed under the BSD 3-clause license found in the
``LICENSE`` file in the root directory of this source tree.
