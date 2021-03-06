CLI Reference
=============
Although the whole **cxflow** API can be accessed programmatically, the intended way of using it is through the
command line instruments. The design goal is to focus on defining the actual models, datasets etc.
instead of the burdensome code, which just puts all the components together.

With proper installation, the |cxflow| command should become available. The command comes with four basic sub-commands
explained below.

.. |cxflow| raw:: html

   <kbd>cxflow</kbd>

.. code-block:: yaml

   usage: cxflow (train | resume | predict | dataset) [-v] [-o] ...

All the sub-commands share the following arguments:

.. raw:: html

   <table class="docutils option-list" frame="void" rules="none">
   <col class="option" />
   <col class="description" />
   <tbody valign="top">
   <tr><td class="option-group">
   <kbd>--output, -o</kbd></td>
   <td>output directory, defaults to <code class="class="docutils literal"><span class="pre">./log</span></code></td></tr>
   <tr><td class="option-group">
   <kbd>--verbose, -v</kbd></td>
   <td>increase verbosity level to <code class="class="docutils literal"><span class="pre">DEBUG</span></code></td></tr>
   </tbody>
   </table>

cxflow train
------------
.. argparse::
   :ref: cxflow.cli.get_cxflow_arg_parser
   :prog: cxflow
   :path: train

cxflow resume
-------------
.. argparse::
   :ref: cxflow.cli.get_cxflow_arg_parser
   :prog: cxflow
   :path: resume

cxflow predict
--------------
.. argparse::
   :ref: cxflow.cli.get_cxflow_arg_parser
   :prog: cxflow
   :path: predict

cxflow dataset
--------------
.. argparse::
   :ref: cxflow.cli.get_cxflow_arg_parser
   :prog: cxflow
   :path: dataset

cxflow ls
--------------
.. argparse::
   :ref: cxflow.cli.get_cxflow_arg_parser
   :prog: cxflow
   :path: ls


cxflow prune
--------------
.. argparse::
   :ref: cxflow.cli.get_cxflow_arg_parser
   :prog: cxflow
   :path: prune
