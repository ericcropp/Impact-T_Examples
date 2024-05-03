This repo contains 3 examples:
1) Running Impact-T on the FACET-II lattice
2) Running a parameter scan on the same setup using Xopt
3) Pulling data from the parameter scan and comparing to actual data taken on the machine

Each folder has a standalone example, and for the third example, files can be loaded from the links in the repo, assuming SLAC s3df access.

Note that packages as listed in the import statements will need to be installed.  Additionally, Impact needs to be installed, as shown here: 

https://github.com/impact-lbl/IMPACT-T/blob/master/README.md
conda install -c conda-forge impact-t
conda install -c conda-forge impact-t=*=mpi_openmpi*

File paths must be updated to store files in the proper locations.  

Finally, to work without changing the mpi_command, you must have access to s3df and the beamphysics repo.  Your account may have different access permissions.  The mpi_run command will need to be updated to reflect these different permissions.



