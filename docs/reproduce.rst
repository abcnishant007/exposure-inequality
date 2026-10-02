Reproducing results
===================

What is and is not included
---------------------------

Raw inputs (licensed Replica mobility data, PurpleAir data, GHAP baseline
PM2.5) are not distributed; see ``DATA_AVAILABILITY.md`` in the repository
root. Full city-scale estimates and the paper results therefore require
obtaining those inputs. Derived paper-result tables are also excluded from this
public code release because they are generated from licensed and third-party
inputs.

Steps
-----

1. Create the environment::

      conda env create -f environment.yml
      conda activate odmatrix

2. Copy and edit the config::

      cp config.yaml.example config.yaml

   Set the study-area GeoJSON, the activity data path, the PM2.5 input paths
   and the output roots. The example config is not runnable until these are
   replaced.

3. Run the pipeline::

      python main_pipeline.py --config config.yaml

   See :doc:`pipeline` for what each module does.

4. Run the public unit tests (no private data needed)::

      pytest tests -m "not private_data"
