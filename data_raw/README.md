# Raw data contract

This directory is for immutable TNTP, OSM-derived, and SUMO inputs. Never write
generated scenarios here. The TNTP loaders are in `data_raw/tntp.py`.

The repository intentionally does not fabricate Anaheim, Winnipeg, Eastern
Massachusetts, or SUMO networks. Put licensed/downloaded source files here, record their
hashes, and process them into `data_processed/generated/` only after their topology
split has been assigned.

